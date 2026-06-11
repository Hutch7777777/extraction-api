"""
Unit tests for services/detection_normalization.py — the read-side choke
point for corner class naming (audit finding T-1) and real-dimension
derivation (audit finding T-2).

Run with:  python3 -m unittest discover tests -v
"""

import unittest
from unittest.mock import Mock

from services.detection_normalization import (
    normalize_detection_class,
    parse_content_dimensions_ft,
    derive_real_dimensions_ft,
    OUTSIDE_CORNER,
    INSIDE_CORNER,
)


class TestNormalizeDetectionClass(unittest.TestCase):
    """All corner spellings collapse to one canonical name."""

    def test_fresh_import_spellings(self):
        # bluebeam_fresh_import_service + intelligent_analysis_service write these
        self.assertEqual(normalize_detection_class('corner_outside'), OUTSIDE_CORNER)
        self.assertEqual(normalize_detection_class('corner_inside'), INSIDE_CORNER)

    def test_roundtrip_import_spellings(self):
        # bluebeam_import_service CLASS_NAME_MAPPING writes these
        self.assertEqual(normalize_detection_class('outside_corner'), OUTSIDE_CORNER)
        self.assertEqual(normalize_detection_class('inside_corner'), INSIDE_CORNER)

    def test_bare_corner_is_outside(self):
        # Old import maps Bluebeam subject 'Corner' to bare 'corner';
        # fresh import treats the same subject as an outside corner
        self.assertEqual(normalize_detection_class('corner'), OUTSIDE_CORNER)

    def test_case_and_space_variants(self):
        self.assertEqual(normalize_detection_class('Outside Corner'), OUTSIDE_CORNER)
        self.assertEqual(normalize_detection_class('CORNER_INSIDE'), INSIDE_CORNER)
        self.assertEqual(normalize_detection_class('  inside corner  '), INSIDE_CORNER)
        self.assertEqual(normalize_detection_class('Corner Outside'), OUTSIDE_CORNER)

    def test_non_corner_classes_pass_through_cleaned(self):
        self.assertEqual(normalize_detection_class('window'), 'window')
        self.assertEqual(normalize_detection_class('Garage Door'), 'garage_door')
        self.assertEqual(normalize_detection_class('Exterior Wall'), 'exterior_wall')

    def test_empty_and_none(self):
        self.assertEqual(normalize_detection_class(None), '')
        self.assertEqual(normalize_detection_class(''), '')


class TestParseContentDimensionsFt(unittest.TestCase):
    """Dimension parsing from Bluebeam content text."""

    def test_feet_with_apostrophes(self):
        self.assertEqual(parse_content_dimensions_ft("10' x 8'"), (10.0, 8.0))
        self.assertEqual(parse_content_dimensions_ft("10'x8'"), (10.0, 8.0))

    def test_feet_and_inches(self):
        self.assertEqual(parse_content_dimensions_ft('3\'-6" x 5\'-0"'), (3.5, 5.0))

    def test_inches_only(self):
        self.assertEqual(parse_content_dimensions_ft('36" x 60"'), (3.0, 5.0))

    def test_ft_keyword(self):
        self.assertEqual(parse_content_dimensions_ft('12 ft x 8.5 ft'), (12.0, 8.5))

    def test_unicode_multiplication_sign(self):
        self.assertEqual(parse_content_dimensions_ft("10' × 8'"), (10.0, 8.0))

    def test_bare_numbers_rejected(self):
        # Ambiguous (feet vs inches) — must not guess
        self.assertIsNone(parse_content_dimensions_ft('36 x 60'))

    def test_non_dimension_content_rejected(self):
        self.assertIsNone(parse_content_dimensions_ft('141 sf'))
        self.assertIsNone(parse_content_dimensions_ft('12.5 lf'))
        self.assertIsNone(parse_content_dimensions_ft('6'))
        self.assertIsNone(parse_content_dimensions_ft(''))
        self.assertIsNone(parse_content_dimensions_ft(None))


class TestDeriveRealDimensionsFt(unittest.TestCase):
    """Fallback chain: stored → pixel×scale → content → warn+zero."""

    def test_stored_columns_win(self):
        det = {'real_width_ft': 4.5, 'real_height_ft': 6.0,
               'pixel_width': 999, 'pixel_height': 999}
        page = {'scale_ratio': 48, 'dpi': 200}
        width, height, source = derive_real_dimensions_ft(det, page)
        self.assertEqual((width, height, source), (4.5, 6.0, 'stored'))

    def test_pixel_scale_derivation(self):
        # 500px * (48 / 200dpi) = 120 in = 10 ft; 250px → 5 ft
        det = {'pixel_width': 500, 'pixel_height': 250, 'markup_type': 'rect'}
        page = {'scale_ratio': 48, 'dpi': 200}
        width, height, source = derive_real_dimensions_ft(det, page)
        self.assertEqual((width, height, source), (10.0, 5.0, 'pixel_scale'))

    def test_pixel_scale_uses_default_dpi_when_page_has_none(self):
        # config.DEFAULT_DPI is 200
        det = {'pixel_width': 500, 'pixel_height': 250}
        page = {'scale_ratio': 48}
        width, height, source = derive_real_dimensions_ft(det, page)
        self.assertEqual((width, height, source), (10.0, 5.0, 'pixel_scale'))

    def test_point_markup_skips_pixel_derivation(self):
        # A Count markup's 16px box is a marker, not a measurement —
        # even with a valid scale it must not produce 0.32 ft "dimensions"
        warn = Mock()
        det = {'id': 'det-1', 'class': 'corner_outside', 'markup_type': 'point',
               'pixel_width': 16, 'pixel_height': 16}
        page = {'scale_ratio': 48, 'dpi': 200}
        width, height, source = derive_real_dimensions_ft(det, page, warn=warn)
        self.assertEqual((width, height, source), (0.0, 0.0, None))
        warn.assert_called_once()

    def test_tiny_box_without_markup_type_treated_as_point(self):
        # intelligent_analysis corner markers are 20x20 px; older rows may
        # lack markup_type entirely
        warn = Mock()
        det = {'id': 'det-2', 'class': 'corner_outside',
               'pixel_width': 20, 'pixel_height': 20}
        page = {'scale_ratio': 48, 'dpi': 200}
        width, height, source = derive_real_dimensions_ft(det, page, warn=warn)
        self.assertEqual(source, None)
        warn.assert_called_once()

    def test_content_fallback_when_scale_missing(self):
        # Fresh-import pages never set scale_ratio (audit finding T-3/T-4)
        det = {'pixel_width': 500, 'pixel_height': 250,
               'bluebeam_content': "10' x 8'"}
        page = {'dpi': 150}  # no scale_ratio
        width, height, source = derive_real_dimensions_ft(det, page)
        self.assertEqual((width, height, source), (10.0, 8.0, 'content'))

    def test_warning_names_detection_id_when_nothing_derivable(self):
        warn = Mock()
        det = {'id': 'det-abc-123', 'class': 'window',
               'pixel_width': 500, 'pixel_height': 250,
               'bluebeam_content': '141 sf'}
        width, height, source = derive_real_dimensions_ft(det, {}, warn=warn)
        self.assertEqual((width, height, source), (0.0, 0.0, None))
        warn.assert_called_once()
        message = warn.call_args[0][0]
        self.assertIn('det-abc-123', message)
        self.assertIn('window', message)

    def test_missing_page_record(self):
        warn = Mock()
        det = {'id': 'det-3', 'pixel_width': 100, 'pixel_height': 100}
        width, height, source = derive_real_dimensions_ft(det, None, warn=warn)
        self.assertEqual((width, height, source), (0.0, 0.0, None))
        warn.assert_called_once()

    def test_string_db_values_coerced(self):
        # PostgREST can hand back numerics as strings
        det = {'pixel_width': '500', 'pixel_height': '250'}
        page = {'scale_ratio': '48', 'dpi': '200'}
        width, height, source = derive_real_dimensions_ft(det, page)
        self.assertEqual((width, height, source), (10.0, 5.0, 'pixel_scale'))


if __name__ == '__main__':
    unittest.main()
