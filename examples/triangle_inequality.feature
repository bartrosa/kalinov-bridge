# language: en
@analysis
Feature: Triangle inequality for real numbers
  Bounding |x + y| by |x| + |y|.

  Scenario: bound on sums
    Given real numbers x and y
    When we consider their absolute values
    Then the following inequality cases apply
      | form of left member | form of right member   |
      | abs(x + y)          | abs(x) + abs(y)        |
