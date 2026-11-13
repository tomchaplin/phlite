<script lang="ts">
  import * as THREE from 'three';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import { onMount } from 'svelte';
  import { dimColour, type Bar } from '../../lib/types';
  import { countUpTo } from './data';

  let {
    points,
    t,
    bar,
    edges,
  }: {
    points: number[][];
    t: number | null;
    bar: Bar | null;
    edges: { pairs: Uint32Array; lengths: Float64Array } | null;
  } = $props();

  let width = $state(500);
  const height = $derived(Math.round(width * 0.8));
  let host: HTMLDivElement;
  let renderer: THREE.WebGLRenderer | null = null;
  let camera: THREE.PerspectiveCamera;
  let controls: OrbitControls;
  const scene = new THREE.Scene();
  const groups = { points: new THREE.Group(), complex: new THREE.Group(), rep: new THREE.Group() };
  Object.values(groups).forEach((g) => scene.add(g));
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const light = new THREE.DirectionalLight(0xffffff, 1.4);
  light.position.set(3, 5, 4);
  scene.add(light);
  const EDGE_LIMIT = 30000;

  const render = () => renderer?.render(scene, camera);

  function clear(group: THREE.Group) {
    for (const child of [...group.children]) {
      group.remove(child);
      const c = child as THREE.Mesh;
      c.geometry?.dispose();
      (c.material as THREE.Material | undefined)?.dispose();
    }
  }

  const centre = $derived.by(() => {
    const c = [0, 0, 0];
    for (const p of points) for (let k = 0; k < 3; k++) c[k] += p[k] / Math.max(points.length, 1);
    let r = 0;
    for (const p of points) r = Math.max(r, Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]));
    return { c, r: r || 1 };
  });
  const v = (i: number) => new THREE.Vector3(points[i][0], points[i][1], points[i][2]);

  onMount(() => {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(devicePixelRatio || 1);
    host.appendChild(renderer.domElement);
    camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.addEventListener('change', render);
    return () => {
      controls.dispose();
      renderer?.dispose();
      renderer = null;
    };
  });

  // The camera, when the points change
  $effect(() => {
    const { c, r } = centre;
    if (!renderer) return;
    camera.position.set(c[0] + r * 1.6, c[1] + r * 1.2, c[2] + r * 2.4);
    controls.target.set(c[0], c[1], c[2]);
    controls.update();
  });

  // The points
  $effect(() => {
    clear(groups.points);
    const { r } = centre;
    const geometry = new THREE.SphereGeometry(r * (points.length > 500 ? 0.008 : 0.014), 8, 6);
    const material = new THREE.MeshLambertMaterial({ color: 0x555b66 });
    const mesh = new THREE.InstancedMesh(geometry, material, points.length);
    const m = new THREE.Matrix4();
    points.forEach((p, i) => mesh.setMatrixAt(i, m.makeTranslation(p[0], p[1], p[2])));
    groups.points.add(mesh);
    render();
  });

  // The Rips edges at t
  $effect(() => {
    clear(groups.complex);
    if (t !== null && edges) {
      const count = Math.min(countUpTo(edges.lengths, t), EDGE_LIMIT);
      const positions = new Float32Array(count * 6);
      for (let k = 0; k < count; k++) {
        const a = points[edges.pairs[2 * k]];
        const b = points[edges.pairs[2 * k + 1]];
        positions.set([a[0], a[1], a[2], b[0], b[1], b[2]], 6 * k);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      groups.complex.add(
        new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0x8a919c, transparent: true, opacity: count > 3000 ? 0.18 : 0.4 })),
      );
    }
    render();
  });

  // The representative (tubes for edges, so they show up thick), and the killing simplex
  $effect(() => {
    clear(groups.rep);
    if (bar?.rep) {
      const colour = new THREE.Color(dimColour(bar.dim));
      const { r } = centre;
      const material = new THREE.MeshLambertMaterial({ color: colour });
      const faces: number[] = [];
      for (const [, cell] of bar.rep) {
        if (cell.length === 1) {
          const s = new THREE.Mesh(new THREE.SphereGeometry(r * 0.03, 12, 8), material);
          s.position.copy(v(cell[0]));
          groups.rep.add(s);
        } else if (cell.length === 2) {
          const a = v(cell[0]);
          const b = v(cell[1]);
          const tube = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.01, r * 0.01, a.distanceTo(b), 6), material);
          tube.position.copy(a).add(b).multiplyScalar(0.5);
          tube.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
          groups.rep.add(tube);
        } else {
          for (const i of cell.slice(0, 3)) faces.push(...points[i]);
        }
      }
      if (faces.length) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(faces), 3));
        geometry.computeVertexNormals();
        groups.rep.add(
          new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ color: colour, transparent: true, opacity: 0.55, side: THREE.DoubleSide })),
        );
      }
    }
    render();
  });

  $effect(() => {
    if (!renderer) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    render();
  });
</script>

<div class="view" bind:clientWidth={width} style="height: {height}px">
  <div bind:this={host} class="host"></div>
  <span class="hint">drag to rotate · scroll to zoom</span>
</div>

<style>
  .host { width: 100%; height: 100%; }
  .hint { position: absolute; right: 0.5rem; bottom: 0.3rem; font-size: 0.75rem; color: var(--muted); pointer-events: none; }
</style>
