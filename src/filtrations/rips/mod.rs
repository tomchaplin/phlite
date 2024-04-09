use ordered_float::NotNan;

use crate::matrices::BasisElement;

pub mod cohomology;
pub mod homology;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RipsIndex(usize);

impl BasisElement for RipsIndex {}

impl RipsIndex {
    /// Produces from least significant to most significant
    pub fn to_vec(self, n_points: usize) -> Vec<usize> {
        let mut vec = vec![];
        let mut working = self.0;
        while working > 0 {
            let remainder = working.rem_euclid(n_points + 1);
            vec.push(remainder - 1);
            working = working.div_euclid(n_points + 1);
        }
        vec
    }

    pub fn from_indices(indices: impl Iterator<Item = usize>, n_points: usize) -> Option<Self> {
        let inner = indices
            .enumerate()
            .map(|(i, coeff)| (n_points + 1).pow(i as u32) * (coeff + 1))
            .sum();
        if inner > 0 {
            Some(Self(inner))
        } else {
            None
        }
    }

    pub fn dimension(self, n_points: usize) -> usize {
        self.to_vec(n_points).len() - 1
    }
}

fn add_cofacets<F, O>(
    bases: &mut Vec<Vec<(O, RipsIndex)>>,
    simplex_as_vec: &mut Vec<usize>,
    fil_value: NotNan<f64>,
    distances: &Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    functor: &F,
) where
    F: Fn(NotNan<f64>) -> O,
{
    let n_vertices = distances.len();
    let max_element = simplex_as_vec.last().unwrap();
    let prev_dimension = simplex_as_vec.len() - 1;
    let new_dimension = prev_dimension + 1;
    if *max_element == n_vertices - 1 {
        // Cannot extend
        return;
    }
    for v in (max_element + 1)..n_vertices {
        // Compute filtration value
        let new_filtration_value = simplex_as_vec
            .iter()
            .map(|w| distances[*w][v])
            .max()
            .unwrap();
        let new_filtration_value = new_filtration_value.max(fil_value);
        // Build up coface
        simplex_as_vec.push(v);
        // Insert into basis
        let index = RipsIndex::from_indices(simplex_as_vec.iter().copied(), n_vertices).unwrap();
        bases[new_dimension].push((functor(new_filtration_value), index));
        // Recurse to cofacets
        if new_dimension < max_dim {
            add_cofacets(
                bases,
                simplex_as_vec,
                new_filtration_value,
                distances,
                max_dim,
                functor,
            );
        }
        // Remove v ready for next coface
        simplex_as_vec.pop();
    }
}

pub fn build_rips_bases<F, O>(
    distances: &Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    functor: F,
) -> Vec<Vec<(O, RipsIndex)>>
where
    F: Fn(NotNan<f64>) -> O,
    O: Clone + Ord,
{
    let mut bases = vec![vec![]; max_dim + 1];

    let n_points = distances.len();
    // Build up the basis
    for v in 0..(distances.len()) {
        let mut simplex = vec![v];
        bases[0].push((
            functor(NotNan::new(0.0_f64).unwrap()),
            RipsIndex::from_indices(vec![v].into_iter(), n_points).unwrap(),
        ));
        add_cofacets(
            &mut bases,
            &mut simplex,
            NotNan::new(0 as f64).unwrap(),
            &distances,
            max_dim,
            &functor,
        );
    }

    for dim in 0..=max_dim {
        // Sort by filtration value then index
        bases[dim].sort_unstable();
    }
    bases
}

pub fn max_pairwise_distance(
    vertices: &Vec<usize>,
    distances: &Vec<Vec<NotNan<f64>>>,
) -> NotNan<f64> {
    let n_vertices = vertices.len();
    (0..n_vertices)
        .flat_map(|i| (0..n_vertices).map(move |j| (i, j)))
        .map(|(i, j)| {
            let v_i = vertices[i];
            let v_j = vertices[j];
            distances[v_i][v_j]
        })
        .max()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use crate::filtrations::rips::RipsIndex;

    #[test]
    fn test_rips_index() {
        let vec = vec![5, 12, 7];
        let index = RipsIndex::from_indices(vec.iter().copied(), 20).unwrap();
        let as_vec = index.to_vec(20);
        assert_eq!(vec, as_vec);

        // Preserves trailing 0
        let vec = vec![5, 12, 0];
        let index = RipsIndex::from_indices(vec.iter().copied(), 20).unwrap();
        let as_vec = index.to_vec(20);
        assert_eq!(vec, as_vec);
    }
}
