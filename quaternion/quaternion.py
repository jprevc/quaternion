import numpy as np


class Quaternion:
    """
    Quaternion Class. Provides a simple and efficient way to work with quaternions.

    There are three ways to initialize a quaternion.
    1) With four distinct values
    >>>Quaternion(1, -2, 0.343, 4.3232)
    Quaternion(1, -2, 0.343, 4.3232)

    2) With one value as list
    >>>Quaternion([1, -2, 0.343, 4.3232])
    Quaternion(1, -2, 0.343, 4.3232)

    3) With one float value as real part and list as complex part
    >>>Quaternion([1, [-2, 0.343, 4.3232])
    Quaternion(1, -2, 0.343, 4.3232)
    """
    def __init__(self, *values):

        # switch according to the initialization type
        if len(values) == 4:
            # type 1 initialization
            self.values = list(values)
            return
        elif len(values) == 1:
            # type 2 initialization
            if type(values[0]) in [tuple, list, np.ndarray]:
                if len(values[0]) == 4:
                    self.values = list(values[0])
                    return
        elif len(values) == 2:
            # type 3 initialization
            if type(values[0]) in [float, int]\
               and type(values[1]) in [tuple, list, np.ndarray]:
                if len(values[1]) == 3:
                    self.values = list([values[0]] + list(values[1]))
                    return

        # raise ValueError in case initialization parameters were not in correct format
        raise ValueError("Quaternion takes exactly 4 values.")

    def __mul__(self, other):
        """ Called when Quaternion is used in multiplication operation """

        if type(other) == Quaternion:
            # handle quaternion multiplication
            w0, x0, y0, z0 = self.values
            w1, x1, y1, z1 = other.values

            mul_res = [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]

            return Quaternion(mul_res)

        else:
            # handle scalar multiplication
            return Quaternion([val * other for val in self.values])

    def __rmul__(self, other):
        """
        Called when Quaternion is used in multiplication operation and
        is used on the right side of the operand.
        """
        return Quaternion([val * other for val in self.values])

    def __add__(self, other):
        """ Called when Quaternion is used in addition operation """

        w0, x0, y0, z0 = self.values

        if type(other) == Quaternion:
            # handle quaternion addition
            w1, x1, y1, z1 = other.values

            return Quaternion([w0+w1, x0+x1, y0+y1, z0+z1])

        else:
            # handle scalar addition
            return Quaternion([w0 + other, x0, y0, z0])

    def __sub__(self, other):
        """ Called when Quaternion is used in subtraction operation """

        w0, x0, y0, z0 = self.values

        if type(other) == Quaternion:
            # handle quaternion subtraction
            w1, x1, y1, z1 = other.values

            return Quaternion([w0-w1, x0-x1, y0-y1, z0-z1])

        else:
            # handle scalar subtraction
            return Quaternion([w0-other, x0, y0, z0])

    def __truediv__(self, other):
        """ Called when Quaternion is used in division operation """
        if type(other) == Quaternion:
            # division with quaternion is not allowed
            raise ValueError("Division with quaternion is not allowed.")
        else:  # handle scalar division
            return Quaternion([x / other for x in self.values])

    def __eq__(self, other):
        """ Called when Quaternion is used in equality check """

        if type(other) != Quaternion:
            raise ValueError("Equality check can only be performed on Quaternion and other Quaternion.")

        return self.values == other.values

    def __ne__(self, other):
        """ Called when Quaternion is used in inequality check """

        if type(other) != Quaternion:
            raise ValueError("Inequality check can only be performed on Quaternion and other Quaternion.")

        return self.values != other.values

    def __getitem__(self, item):
        """ Called when Quaternion is indexed """
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __repr__(self):
        """ Returns string representation of Quaternion """
        return "Quaternion(" + str(self.values)[1:-1] + ")"

    def __neg__(self):
        """ Called when minus (-) sign is placed before Quaternion """
        return Quaternion([-x for x in self.values])

    def __abs__(self):
        """Called when abs() function is called on Quaternion"""
        return Quaternion([abs(x) for x in self.values])

    @property
    def norm(self):
        """ Returns norm of quaternion which is square root of the sum of its squared values. """
        return np.sqrt(np.sum(np.array(self.values) ** 2))

    @property
    def rot_mat(self):
        """
        Returns rotational matrix which corresponds to the Quaternion

        :return: Rotational matrix
        :rtype: np.array
        """
        qr, qi, qj, qk = self.values

        s = 1 / (self.norm**2.0)

        return np.array([[1 - 2*s*(qj**2.0 + qk**2.0), 2*s*(qi*qj - qk*qr),         2*s*(qi*qk + qj*qr)],
                         [2*s*(qi*qj + qk*qr),         1 - 2*s*(qi**2.0 + qk**2.0), 2*s*(qj*qk - qi*qr)],
                         [2*s*(qi*qk - qj*qr),         2*s*(qj*qk + qi*qr),         1 - 2*s*(qi**2.0 + qj**2.0)]])

    @property
    def euler(self):
        """
        Returns Euler angles in terms of roll, pitch and yaw, which
        correspond to rotations around x, y and z axes, respectively.

        :return: Euler angles in radians (roll, pitch, yaw)
        :rtype: tuple
        """

        q0, q1, q2, q3 = self.values
        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2.0 + q2**2.0))
        pitch = np.arcsin(2*(q0*q2 - q3*q1))
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2.0 + q3**2.0))

        return roll, pitch, yaw

    @property
    def conj(self):
        """
        Returns conjugated Quaternion, which is Quaternion with inversed complex part.

        :return: Conjugated quaternion
        :rtype: Quaternion
        """
        return Quaternion([self.values[0]] + [-x for x in self.values[1:]])

    @property
    def real(self):
        """ Returns real part of quaternion. """
        return self.values[0]

    @property
    def imag(self):
        """ Returns imaginary part of quaternion as a list. """
        return self.values[1:]

def quaternion_from_euler(roll, pitch, yaw):
    """
    Returns a quaternion which corresponds to euler angles, roll, pitch and yaw.

    :param roll: Euler roll angle in radians.
    :type roll: float

    :param pitch: Euler pitch angle in radians.
    :type pitch: float

    :param yaw: Euler yaw angle in radians.
    :type yaw: float

    :return: Quaternion, which corresponds input to euler angles.
    :rtype: Quaternion
    """

    q0 = Quaternion([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
    q1 = Quaternion([np.cos(roll/2), 0, np.sin(roll/2), 0])
    q2 = Quaternion([np.cos(pitch/2), np.sin(pitch/2), 0, 0])

    return q0*q1*q2
