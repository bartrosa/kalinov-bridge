# language: en
@forthel
Feature: Naproche / ForTheL demo harness

  Scenario: bracket marker on the step line
    Given a lexicon for arithmetic
    When we assert the following claim in ForTheL
    Then [ForTheL] Let n be a natural number. Then n = n.

  Scenario: doc-string body tagged as ftl
    Given background vocabulary
    When the formal statement lives in a doc string
    Then the formalization reads
      """ftl
      Signature. Let x be a real number.
      Theorem. x is equal to x.
      """
