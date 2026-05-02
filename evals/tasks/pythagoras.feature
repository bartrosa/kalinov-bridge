# language: en
@geometry
Feature: Pythagorean theorem in the Euclidean plane
  Relating side lengths of a right triangle.

  Scenario: side lengths of a right triangle
    Given a right triangle with legs a, b and hypotenuse c
    When the triangle lies in ℝ² with the right angle between a and b
    Then the relation holds
      """tex
      a^2 + b^2 = c^2
      """
