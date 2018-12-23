import unittest
from quaternion import Quaternion
import numpy as np


class TestQuaternion(unittest.TestCase):

    def test_init_no_input_raises_error(self):
        with self.assertRaises(ValueError):
            q = Quaternion()

    def test_input_length_not_four_raises_value_error_one_value(self):
        with self.assertRaises(ValueError):
            q = Quaternion(3)

    def test_input_is_ndarray_is_valid(self):
        arr = np.array([1,2,3,4])
        q = Quaternion(arr)

    def test_input_is_scalar_and_ndarray_is_valid(self):
        arr = np.array([1,2,3])
        q = Quaternion(-1, arr)

    def test_input_list_is_valid_input(self):
        q = Quaternion([1,2,3,4])

    def test_input_int_and_list_is_valid_input(self):
        q = Quaternion(1, [2,3,4])

    def test_input_float(self):
        q = Quaternion(1.23, -2.43, 3.34, 4.23)
        self.assertEqual(q, Quaternion(1.23, -2.43, 3.34, 4.23))

    def test_input_tuple_is_valid_input(self):
        q = Quaternion([1,2,3,4])

    def test_input_length_not_four_raises_value_error_five_values(self):
        with self.assertRaises(ValueError):
            q = Quaternion(1,2,3,5,2)

    def test_equality_equal_quaternions(self):
        q1 = Quaternion(-1, 2, -3, 4)
        q2 = Quaternion(-1, 2, -3, 4)

        self.assertTrue(q1 == q2)

    def test_equality_one_different(self):
        q1 = Quaternion(-1, 2, -3, 4)
        q2 = Quaternion(1, 2, -3, 4)

        self.assertFalse(q1 == q2)

    def test_inequality_equal_quaternions(self):
        q1 = Quaternion(-1, 2, -3, 4)
        q2 = Quaternion(-1, 2, -3, 4)

        self.assertFalse(q1 != q2)

    def test_inequality_one_different(self):
        q1 = Quaternion(-1, 2, -3, 4)
        q2 = Quaternion(1, 2, -3, 4)

        self.assertTrue(q1 != q2)

    def test_conjugate(self):
        q1 = Quaternion(1, 2, -3, 4)
        q2 = Quaternion(1, -2, 3, -4)

        self.assertEqual(q1.conj, q2)

    def test_conjugate_zero_vector_part(self):
        q1 = Quaternion(1, 0, 0, 0)

        self.assertEqual(q1.conj, q1)

    def test_repr(self):
        q1 = Quaternion(-1, 2, -3, 4)
        self.assertEqual(q1.__repr__(), "Quaternion(-1, 2, -3, 4)")

    def test_multiplication_with_scalar(self):
        q1 = Quaternion(1, -2, 3, 4)
        self.assertEqual(q1 * 5, Quaternion(5, -10, 15, 20))

    def test_multiplication_with_scalar_right_side(self):
        q1 = Quaternion(1, -2, 3, 4)
        self.assertEqual(0.5 * q1, Quaternion(0.5, -1, 1.5, 2))

    def test_multiplication_with_negative_scalar(self):
        q1 = Quaternion(1, -2, 3, 4)
        self.assertEqual(q1 * (-3), Quaternion(-3, 6, -9, -12))

    def test_addition_with_scalar(self):
        q1 = Quaternion(1, -2, 3, 4)
        self.assertEqual(q1 + 3, Quaternion(4, -2, 3, 4))

    def test_addition_with_quaternion(self):
        q1 = Quaternion(1, -2, 3, 4)
        q2 = Quaternion(3, 5, -3, 10)
        self.assertEqual(q1 + q2, Quaternion(4, 3, 0, 14))

    def test_subtraction_with_scalar(self):
        q1 = Quaternion(1, -2, 3, 4)
        self.assertEqual(q1 - 5, Quaternion(-4, -2, 3, 4))

    def test_subtraction_with_quaternion(self):
        q1 = Quaternion(1, -2, 3, 4)
        q2 = Quaternion(3, 5, -3, 10)
        self.assertEqual(q1 - q2, Quaternion(-2, -7, 6, -6))

    def test_division_with_scalar(self):
        q1 = Quaternion(3, -6, 9, -12)
        self.assertEqual(q1 / 3, Quaternion(1, -2, 3, -4))

    def test_division_with_quaternion_raises_error(self):
        with self.assertRaises(ValueError):
            q1 = Quaternion(3, -6, 9, -12)
            q2 = Quaternion(3, -6, 9, -12)
            div = q1 / q2

    def test_minus_sign(self):
        q1 = Quaternion(1, -2, 3, 4)
        self.assertEqual(-q1, Quaternion(-1, 2, -3, -4))

    def test_norm_known_value(self):
        inp_data = [3, -4, 0, -5]
        q1 = Quaternion(inp_data)
        self.assertAlmostEqual(q1.norm, 7.071067, delta=1e-6)

    def test_abs_value(self):
        inp_data = [3, -4, 0, -5]
        q1 = Quaternion(inp_data)
        self.assertEqual(abs(q1), Quaternion(3, 4, 0, 5))

    def test_indexing_one_item(self):
        q = Quaternion(5, -2, 3, 4)
        self.assertTrue(q[0] == 5)

    def test_indexing_slice(self):
        q = Quaternion(5, -2, 3, 4)
        self.assertEqual(q[1:], [-2,3,4])

    def test_set_index(self):
        q = Quaternion(5, -2, 3, 4)
        q[1] = 10
        self.assertEqual(q, Quaternion(5,10,3,4))

    def test_set_index_slice(self):
        q = Quaternion(5, -2, 3, 4)
        q[1:] = [5,6,7]
        self.assertEqual(q, Quaternion(5, 5, 6, 7))

    def test_imag_part_property(self):
        q = Quaternion(5, -2, 3, 4)
        self.assertEqual(q.imag, [-2,3,4])

    def test_real_part_property(self):
        q = Quaternion(5, -2, 3, 4)
        self.assertEqual(q.real, 5)



if __name__ == '__main__':
    unittest.main()



