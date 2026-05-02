# language: en
@physics
@mechanics
Feature: One-dimensional elastic collision
  Conservation of momentum and kinetic energy in 1D.

  Background:
    Given two point masses on a line with masses m₁ and m₂
    And initial velocities v₁ and v₂ along the same axis

  Scenario: momentum is conserved
    When the masses collide elastically
    Then m₁ v₁ + m₂ v₂ equals m₁ v₁′ + m₂ v₂′
