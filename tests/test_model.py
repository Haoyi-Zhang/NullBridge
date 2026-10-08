import struct
import unittest

from nullbridge.logical import I32, BOOL, LogicalTable, canonical, bag
from nullbridge.physical import Array, PhysicalTable, InvalidRepresentation, certify, decode


class ModelTests(unittest.TestCase):
    def test_primitive_certificate(self):
        logical = LogicalTable((("value", I32),), ((12,), (25,)))
        array = Array(I32, 2, 0, (None, struct.pack("<ii", 12, 25)))
        physical = PhysicalTable(logical.fields, ((array,),))
        self.assertEqual(certify(logical, physical)["rows"], 2)

    def test_nullable_primitive(self):
        array = Array(I32, 2, 0, (bytes([1]), struct.pack("<ii", 12, 25)))
        self.assertEqual(decode(array), [12, None])

    def test_owned_primitive_slice(self):
        array = Array(I32, 3, 0, (None, struct.pack("<iii", 12, 25, 40)))
        self.assertEqual(decode(array.slice(1, 1)), [25])

    def test_json_round_trip(self):
        array = Array(I32, 1, 0, (None, struct.pack("<i", 12)))
        self.assertEqual(Array.from_json(array.to_json()), array)

    def test_bool_is_not_integer(self):
        self.assertNotEqual(canonical(True), canonical(1))
        LogicalTable((("flag", BOOL),), ((True,),)).validate()

    def test_multiplicity_is_retained(self):
        self.assertNotEqual(bag([(12,), (12,)]), bag([(12,)]))

    def test_incomplete_owned_buffer_rejected(self):
        logical = LogicalTable((("value", I32),), ((12,),))
        array = Array(I32, 1, 0, (None, b""))
        with self.assertRaises(InvalidRepresentation):
            certify(logical, PhysicalTable(logical.fields, ((array,),)))

    def test_value_mismatch_rejected(self):
        logical = LogicalTable((("value", I32),), ((12,),))
        array = Array(I32, 1, 0, (None, struct.pack("<i", 25)))
        with self.assertRaises(InvalidRepresentation):
            certify(logical, PhysicalTable(logical.fields, ((array,),)))


if __name__ == "__main__":
    unittest.main()
