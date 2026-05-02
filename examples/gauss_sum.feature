# language: en
@math
@arithmetic
Feature: Sum of the first n natural numbers
  A closed form for the discrete sum used in many induction exercises.

  @step_series
  Scenario Outline: closed form for 1 + 2 + … + n
    Given a positive integer <n>
    When we apply the identity for the arithmetic series
    Then the sum should equal <expected>

    @first_rows
    Examples:
      | n  | expected |
      | 1  | 1        |
      | 5  | 15       |
      | 10 | 55       |
