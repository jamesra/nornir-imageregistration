import numpy as np


class DistanceSquaredArray(np.ndarray):
    '''
    This returns the distance from any index to the center specified at creation, however it
    uses a function to calculate the distance instead of storing the values in memory
    '''

    def __new__(cls, shape, *args, **kwargs):
        obj = np.empty(shape, dtype=np.float32).view(cls)
        ndims = len(shape)
        obj._center = np.zeros(len(shape))
        obj._distances = []
        obj._odd_dimension = np.zeros(len(shape), dtype=bool)

        for i, dim_size in enumerate(shape):
            is_odd_shape = dim_size % 2 > 0
            obj._odd_dimension[i] = is_odd_shape
            center = dim_size / 2.0
            half_dim_size = center if not is_odd_shape else center - 0.5
            half_dim_size = int(half_dim_size)
            obj._center[i] = center

            if is_odd_shape:
                dist_values_squared = np.linspace(-half_dim_size, half_dim_size, num=dim_size)
            else:
                dist_values_squared = np.linspace(-(half_dim_size - 0.5), half_dim_size - 0.5, num=dim_size)

            dist_values_squared **= 2
            obj._distances.append(dist_values_squared)
        return obj

    def __getitem__(self, index):
        # print(self._distances)
        # print(f'{index}')
        ndims = len(self.shape)
        output = []
        for i, slice in enumerate(index):
            a = self._distances[i][slice]
            # b = np.broadcast_to(a, )
            if isinstance(a, np.ndarray):
                dim_reshape = [1] * ndims
                dim_reshape[i] = len(a)
                reshaped_a = a.reshape(dim_reshape)
                # print(f'i: {i} -> {a} -> res: {dim_reshape} -> {reshaped_a}')
                output.append(reshaped_a)
            else:
                output.append(a)

        o = np.broadcast_arrays(*output)
        sum = np.sum(o, axis=0)
        return sum


class DistanceArray(DistanceSquaredArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(result, np.ndarray):
            np.sqrt(result, out=result)
            return result
        else:
            return np.sqrt(result)


if __name__ == '__main__':
    da = DistanceSquaredArray((4, 3))
    print(f'distance: {da}')
    print(f'distance: {da[1:3, 0:3]}')
    print(f'distance: {da[1, 0:3]}')
    print(f'distance: {da[:, 0:3]}')

    da = DistanceArray((8, 7))
    print(f'distance: {da}')
    print(f'distance: {da[1:3, 0:3]}')
    print(f'distance: {da[1, 0:3]}')
    print(f'distance: {da[:, 0:3]}')
