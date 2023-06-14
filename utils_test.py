"""Unit tests for utils.py."""

import unittest

from utils import *

class UtilsTest(unittest.TestCase):

    def test_load_annotation(self):
        annotation_string = " [{\"acronym\": \"\", \"applications\": [\"photocatalyst\"], \"name\": \"\", \"formula\": \"H1-2xPtxLaNb2O7\", \"structure_or_phase\": [\"perovskite\"], \"description\": [\"layered\", \"intercalated Pt2+\"]}, {\"acronym\": \"\", \"applications\": [\"photocatalyst\"], \"name\": \"\", \"formula\": \"RbLaNb2O7\", \"structure_or_phase\": [\"\"], \"description\": [\"\"]}, {\"acronym\": \"\", \"applications\": [\"photocatalyst\"], \"name\": \"Pt-deposited RbLaNb2O7\", \"formula\": \"\", \"structure_or_phase\": [\"\"], \"description\": [\"\"]}, {\"acronym\": \"\", \"applications\": [\"photocatalyst\"], \"name\": \"\", \"formula\": \"HLaNb2O7\", \"structure_or_phase\": [\"\"], \"description\": [\"\"]}]\n\nEND\n\n"

        stop_sequence = "\n\nEND\n\n"

        annotation = load_annotation(annotation_string, stop_sequence=stop_sequence)

        gold_standard_annotation = [{'acronym': '', 'applications': ['photocatalyst'], 'name': '', 'formula': 'H1-2xPtxLaNb2O7', 'structure_or_phase': ['perovskite'], 'description': ['layered', 'intercalated Pt2+']}, {'acronym': '', 'applications': ['photocatalyst'], 'name': '', 'formula': 'RbLaNb2O7', 'structure_or_phase': [''], 'description': ['']}, {'acronym': '', 'applications': ['photocatalyst'], 'name': 'Pt-deposited RbLaNb2O7', 'formula': '', 'structure_or_phase': [''], 'description': ['']}, {'acronym': '', 'applications': ['photocatalyst'], 'name': '', 'formula': 'HLaNb2O7', 'structure_or_phase': [''], 'description': ['']}]

        self.assertEqual(annotation, gold_standard_annotation)
