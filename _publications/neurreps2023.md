---
title: "Sample Efficient Modeling of Drag Coefficients for Satellites with Symmetry"
collection: publications
permalink: /publication/neurips2023
excerpt: 
date: 2023-12-01
venue: "NeurIPS 2023 Workshop on Symmetry and Geometry in Neural Representations"
paperurl: "https://openreview.net/forum?id=u7r2160QiP"
citation: Neel Sortur, Linfeng Zhao, Robin Walters. Sample Efficient Modeling of Drag Coefficients for Satellites with Symmetry. In <i>NeurIPS Workshop on Symmetry and Geometry in Neural Representations</i>, 2023.
---

[OpenReview](https://openreview.net/forum?id=u7r2160QiP)

<b>Abstract</b>:
Accurate knowledge of the atmospheric drag coefficient for a satellite in low Earth orbit is crucial to plan an orbit that avoids collisions with other spacecraft, but its calculation has high uncertainty and is very expensive to numerically compute for long-horizon predictions. Previous work has improved coefficient modeling speed with data-driven approaches, but these models do not utilize domain symmetry. This work investigates enforcing the invariance of atmospheric particle deflections off certain satellite geometries, resulting in higher sample efficiency and theoretically more robustness for data-driven methods. We train G-equivariant MLPs to predict the drag coefficient, where G defines invariances of the coefficient across different orientations of the satellite. We experiment on a synthetic dataset computed using the numerical Test Particle Monte Carlo (TPMC) method, where particles are fired at a satellite in the computational domain. We find that our method is more sample and computationally efficient than unconstrained baselines, which is significant because TPMC simulations are extremely computationally expensive.
