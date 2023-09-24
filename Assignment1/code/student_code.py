import numpy as np


def calculate_convolution(image, kernel):
    # rotating kernel with 180 degrees
    kernel = np.rot90(kernel, 2)

    kernel_heigh = int(np.array(kernel).shape[0])
    kernel_width = int(np.array(kernel).shape[1])

    # set kernel matrix to random int matrix
    if (kernel_heigh % 2 != 0) & (kernel_width % 2 != 0):  # make sure that the scale of kernel is odd
        # the scale of result
        conv_heigh = image.shape[0] - kernel.shape[0] + 1
        conv_width = image.shape[1] - kernel.shape[1] + 1
        conv = np.zeros((conv_heigh, conv_width))

        # convolve
        for i in range(int(conv_heigh)):
            for j in range(int(conv_width)):
                result = (image[i:i + kernel_heigh, j:j + kernel_width] * kernel).sum()
                # if(result<0):
                #     resutl = 0
                # elif(result>255):
                #     result = 255
                conv[i][j] = result
    return conv


def my_imfilter(image, filter):
    """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    # TODO: YOUR CODE HERE

    # zero padding
    kernel_half_row = int((filter.shape[0] - 1) / 2)
    kernel_half_col = int((filter.shape[1] - 1) / 2)

    # judge how many channels
    image = np.pad(image, ((kernel_half_row, kernel_half_row), (kernel_half_col, kernel_half_col), (0, 0)),
                   'constant', constant_values=0)

    # if image.shape[2] == 3 or image.shape[2] == 4:
    # if style is png, there will be four channels, but we just need to use the first three
    # if the style is bmp or jpg, there will be three channels
    image_r = image[:, :, 0]
    image_g = image[:, :, 1]
    image_b = image[:, :, 2]
    result_r = calculate_convolution(image_r, filter)
    result_g = calculate_convolution(image_g, filter)
    result_b = calculate_convolution(image_b, filter)
    filtered_image = np.dstack([result_r, result_g, result_b])

    # raise NotImplementedError('`my_imfilter` function in `student_code.py` ' +
    #                           'needs to be implemented')

    ### END OF STUDENT CODE ####
    ############################
    return filtered_image


def create_hybrid_image(image1, image2, filter):
    """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    ############################
    ### TODO: YOUR CODE HERE ###

    raise NotImplementedError('`create_hybrid_image` function in ' +
                              '`student_code.py` needs to be implemented')

    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image



def myHybridImages(lowImage: np.ndarray, lowSigma, highImage: np.ndarray, highSigma):
    # make kernel
    low_kernel = makeGaussianKernel(lowSigma)
    high_kernel = makeGaussianKernel(highSigma)

    # convolve low-pass pictures
    low_image = convolve(lowImage, low_kernel)

    # make high-pass picture
    high_image = (highImage - convolve(highImage, high_kernel))

    # final picture
    # the weights between and final lighting can be changed flexibly
    weight = 1
    weight2 = 1
    adjustment = 0
    hybrid_image = high_image * weight2 + low_image * weight + adjustment
    # hybrid_image = high_image + low_image

    # randomly double check the output
    # print(hybrid_image[11][22][1])
    # print(hybrid_image[44][55][0])
    # print(hybrid_image[357][159][2])

    return hybrid_image
