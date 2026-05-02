# language: en
@math
@inequality
Feature: Arithmetic–geometric mean for two non-negative reals
  Classical AM–GM in ℝ₊.

  @two_variables
  Scenario: AM-GM for a and b
    Given non-negative real numbers a and b
    When we compare their arithmetic and geometric means
    Then (a + b) / 2 ≥ √(a b)
