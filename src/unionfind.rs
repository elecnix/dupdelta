//! Disjoint-set union, for grouping duplicate pairs into classes.
//!
//! # A caveat worth reading before you trust a class
//!
//! Grouping is transitive but similarity is not. If `A` resembles `B`, `B`
//! resembles `C`, and `C` resembles `D`, all four land in one class even when
//! `A` and `D` share nothing at all. On a large tree this can chain hundreds of
//! unrelated units into a single useless "class".
//!
//! Classes are therefore a *navigation aid*, never the finding. The pair — with
//! its measured similarity — is the finding. Report pairs; use classes to group
//! them for reading.

/// Disjoint-set forest with path halving and union by size.
#[derive(Clone, Debug)]
pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    /// Create `n` singleton sets, numbered `0..n`.
    pub fn new(n: usize) -> Self {
        UnionFind { parent: (0..n).collect(), size: vec![1; n] }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Whether there are no elements.
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Representative of `x`'s set.
    ///
    /// # Panics
    /// If `x` is out of range.
    pub fn find(&mut self, mut x: usize) -> usize {
        assert!(x < self.parent.len(), "index {x} out of range for {} elements", self.parent.len());
        while self.parent[x] != x {
            // Path halving: point at the grandparent as we walk.
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Merge the sets containing `a` and `b`.
    ///
    /// Returns `true` if they were distinct sets and are now joined, `false` if
    /// they were already together.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        // Attach the smaller tree under the larger, keeping depth logarithmic.
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        true
    }

    /// Size of the set containing `x`.
    pub fn set_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }

    /// Every set, each as an ascending list of members.
    ///
    /// Sets are ordered by their lowest member, so the output is deterministic.
    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        let mut by_root: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for x in 0..self.parent.len() {
            let root = self.find(x);
            by_root.entry(root).or_default().push(x);
        }
        let mut groups: Vec<Vec<usize>> = by_root.into_values().collect();
        groups.sort_unstable_by_key(|g| g[0]);
        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_forest_has_every_element_in_its_own_set() {
        let mut uf = UnionFind::new(3);
        assert_eq!(uf.len(), 3);
        assert!(!uf.is_empty());
        assert_eq!(uf.groups(), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn an_empty_forest_reports_itself_empty() {
        let mut uf = UnionFind::new(0);
        assert!(uf.is_empty());
        assert_eq!(uf.len(), 0);
        assert!(uf.groups().is_empty());
    }

    #[test]
    fn union_joins_two_sets_and_reports_the_merge() {
        let mut uf = UnionFind::new(2);
        assert!(uf.union(0, 1));
        assert_eq!(uf.find(0), uf.find(1));
    }

    #[test]
    fn union_of_two_elements_already_together_reports_no_merge() {
        let mut uf = UnionFind::new(2);
        uf.union(0, 1);
        assert!(!uf.union(1, 0));
    }

    #[test]
    fn membership_is_transitive() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(1, 2);
        assert_eq!(uf.groups(), vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn set_size_counts_every_member() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(1, 2);
        assert_eq!(uf.set_size(2), 3);
        assert_eq!(uf.set_size(3), 1);
    }

    #[test]
    fn union_by_size_attaches_the_smaller_tree_under_the_larger() {
        // Exercises both branches of the size comparison: first union grows the
        // left root, second presents the larger set on the right.
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(0, 2);
        uf.union(3, 0);
        assert_eq!(uf.set_size(3), 4);
        assert_eq!(uf.groups(), vec![vec![0, 1, 2, 3], vec![4]]);
    }

    #[test]
    fn path_halving_shortens_a_deep_chain() {
        // Build a chain, then prove repeated lookups still resolve correctly.
        let mut uf = UnionFind::new(8);
        for i in 1..8 {
            uf.union(i - 1, i);
        }
        let root = uf.find(7);
        for i in 0..8 {
            assert_eq!(uf.find(i), root);
        }
        assert_eq!(uf.set_size(0), 8);
    }

    #[test]
    fn groups_are_ordered_by_their_lowest_member() {
        let mut uf = UnionFind::new(6);
        uf.union(4, 5);
        uf.union(1, 3);
        assert_eq!(uf.groups(), vec![vec![0], vec![1, 3], vec![2], vec![4, 5]]);
    }

    #[test]
    fn the_forest_clones_and_debugs() {
        let mut uf = UnionFind::new(2);
        uf.union(0, 1);
        let mut copy = uf.clone();
        assert_eq!(copy.groups(), uf.groups());
        assert!(format!("{uf:?}").contains("UnionFind"));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn find_rejects_an_index_that_does_not_exist() {
        UnionFind::new(1).find(5);
    }
}
