import numpy as np


class DistanceArray(np.ndarray):
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
                dist_values_squared = np.linspace(0, half_dim_size, num=half_dim_size + 1)
            else:
                dist_values_squared = np.linspace(0.5, half_dim_size - 0.5, num=half_dim_size)

            #center += ((dim_size % 2) - 1) / 2.0
            #dist_values_squared = np.linspace(-center, center, dim_size)
            dist_values_squared **= 2

            #dim_reshape = [1] * ndims
            #dim_reshape[i] = len(dist_values_squared)

            obj._distances.append(dist_values_squared)
        return obj

    def __getitem__(self, index):
        print(self._distances)
        print(f'{index}')
        value = None
        ndims = len(self.shape)
        desired_shape = [1] * ndims
        output = []
        for i, slice in enumerate(index):
            a = self._distances[i][slice]
            #b = np.broadcast_to(a, )
            if isinstance(a, np.ndarray):
                dim_reshape = [1] * ndims
                dim_reshape[i] = len(a)
                reshaped_a = a.reshape(dim_reshape)
                print(f'i: {i} -> {a} -> res: {dim_reshape} -> {reshaped_a}')
            else:
                reshaped_a = a

            output.append(reshaped_a)

        o = np.broadcast_arrays(*output)
        sum = np.sum(o, axis=0)
        print(f'o: {o} -> sum: {sum}')
            # if value is None:
            #     value = reshaped_a
            # else:
            #     value = np.add(value, reshaped_a)

        return value

if __name__ == '__main__':
    da = DistanceArray((8,7))
    print(f'distance: {da[1:3,0:3]}')
    print(f'distance: {da[1,0:3]}')
    print(f'distance: {da[:,0:3]}')