import numpy as np


# by yunfei: This function is for ground truth existed in dataset CityStreet. Because if you wanna use existing density maps of
# different resolutions, you must make the sum of density maps is the same as previous for sure.

def conv_process(img, pad=0, stride=4, filter_size=4, dim_ordering='tf'):
    # print("Warning. You are using conv_process.")
    # print("Try to use scipy.ndimage.convolve for speed")
    # sys.exit(1)
    # suitable for ndarray data type
    if stride == 1 and filter_size==1:  #make sure return original image.
        return img
    assert img.ndim == 3
    assert stride == filter_size
    assert dim_ordering in ['th', 'tf']
    if dim_ordering == 'th':
        hy_rows = img.shape[1]
        wx_cols = img.shape[2]
        n_channel = img.shape[0]
    elif dim_ordering == 'tf':
        hy_rows = img.shape[0]
        wx_cols = img.shape[1]
        n_channel = img.shape[2]
    assert hy_rows % filter_size == 0
    assert wx_cols % filter_size == 0
    assert n_channel in [1]
    # range_y = range(0, hy_rows + 2 * pad - filter_size + 1, stride)
    # range_x = range(0, wx_cols + 2 * pad - filter_size + 1, stride)
    # range_y = range(0 + 4, hy_rows - 4 + 2 * pad - filter_size + 1, stride)
    range_y = range(0, hy_rows + 2 * pad - filter_size + 1, stride)
    # print(range_y)
    # range_x = range(0 + 4, wx_cols - 4 + 2 * pad - filter_size + 1, stride)
    range_x = range(0, wx_cols + 2 * pad - filter_size + 1, stride)
    # print(range_x)
    output_rows = len(range_y)
    output_cols = len(range_x)
    # print 'output size', output_rows, output_cols
    # new_dim = output_rows * output_cols
    # print('new size is: {} * {}'.format(output_rows, output_cols))
    # print('new dim is: {}'.format(new_dim))
    if dim_ordering == 'th':
        result = np.zeros((n_channel, output_rows, output_cols), dtype=np.single)
    elif dim_ordering == 'tf':
        result = np.zeros((output_rows, output_cols, n_channel), dtype=np.single)
    for index in range(n_channel):
        if dim_ordering == 'th':
            if pad > 0:
                new_data = np.zeros(
                    [hy_rows + 2 * pad, wx_cols + 2 * pad], dtype=np.single)
                new_data[pad:pad + hy_rows, pad:pad + wx_cols] = img[index, ...]
            else:
                new_data = img[index, ...]

            y_ind = 0
            for y in range_y:
                x_ind = 0
                for x in range_x:
                    # print new_data_mat[y:y + filter_size, x:x + filter_size]
                    # print x_ind, y_ind
                    result[index, y_ind, x_ind] = new_data[y:y + filter_size, x:x + filter_size].sum()
                    x_ind += 1
                y_ind += 1
        elif dim_ordering == 'tf':
            if pad > 0:
                new_data = np.zeros(
                    [hy_rows + 2 * pad, wx_cols + 2 * pad], dtype=np.single)
                new_data[pad:pad + hy_rows, pad:pad + wx_cols] = img[..., index]
            else:
                new_data = img[..., index]

            y_ind = 0
            for y in range_y:
                x_ind = 0
                for x in range_x:
                    # print new_data_mat[y:y + filter_size, x:x + filter_size]
                    # print x_ind, y_ind
                    result[y_ind, x_ind, index] = new_data[y:y + filter_size, x:x + filter_size].sum()
                    x_ind += 1
                y_ind += 1
    return result
