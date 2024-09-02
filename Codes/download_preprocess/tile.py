import os
import numpy as np
import rasterio as rio
from Codes.utils.system_ops import makedirs
from rasterio.windows import Window, transform


class make_tiles:
    def __init__(self, tiff_path, band_key_list, tile_output_dir, tile_height=64, tile_width=64, skip_processing=False):
        if not skip_processing:

            makedirs([tile_output_dir])

            # Opening a image file. The tile-ing operation will be done inside the opened image file.
            with rio.open(tiff_path) as tiff:

                # storing image width, height, and nodata for later use
                tiff_height = tiff.height
                tiff_width = tiff.width
                self.nodata = tiff.nodata

                # getting year (+ month) info which will be later used in saving tile name
                if len(os.path.basename(tiff_path).split('.')[0].split('_')[-1]) == 4:  # check for annual data
                    year = os.path.basename(tiff_path).split('.')[0].split('_')[-1]
                    month = None

                else:  # check for monthly data
                    year = os.path.basename(tiff_path).split('.')[0].split('_')[-2]
                    month = os.path.basename(tiff_path).split('.')[0].split('_')[-1]

                # initiating tile number
                tile_no = 1

                # The 1st loop takes across width and the 2nd loop takes across height.
                # The 2 loops together creates a window which is used to create tile from the image.
                for i in range(0, tiff_width, tile_width):
                    for j in range(0, tiff_height, tile_height):
                        if (i + tile_width <= tiff_width) and (
                                j + tile_height <= tiff_height):  # a check to keep the window within the image
                            window = Window(col_off=i, row_off=j, width=tile_width,
                                            height=tile_height)  # the tile window

                            tile_arr = tiff.read(window=window)  # reading the image as an array for the tile window

                            tile_name = f'{year}_tile_{tile_no}' if month is None else f'{year}_{month}_tile_{tile_no}'  # tile name

                            window_transform = transform(window, tiff.transform)  # tiled window's affine transformation

                            # saving the tiled image
                            output_file = os.path.join(tile_output_dir, f'{tile_name}.tif')

                            # in case complete nodata tile detected, exiting the loop into next iteration
                            if make_tiles.is_image_null(tile_arr):
                                continue

                            # a check to see if the tile is completely nodata. In case of an unexpected nodata tile passes
                            # through the if block, the assertion block will raise AssertionError
                            assert not make_tiles.is_image_null(tile_arr), f'All nodata value in tile {tile_no}'

                            # if the % nodata is greater than a threshold, not including those tiles
                            nodata_threshold = 95  # (unit in percent)

                            if any(x > nodata_threshold for x in make_tiles.count_perc_nodata(tile_arr)):
                                continue

                            with rio.open(
                                    output_file,
                                    'w',
                                    driver='GTiff',
                                    height=tile_arr.shape[1],
                                    width=tile_arr.shape[2],
                                    dtype=tile_arr.dtype,
                                    count=tile_arr.shape[0],
                                    crs=tiff.crs,
                                    transform=window_transform,
                                    nodata=self.nodata
                            ) as dst:

                                for id in range(0,
                                                tile_arr.shape[0]):  # looping for each band of the windowed/tiled array
                                    dst.write_band(id + 1, tile_arr[id])
                                    dst.set_band_description(id + 1, band_key_list[id])

                            tile_no += 1

    @staticmethod
    def is_image_null(multiband_img_arr, nodata=-9999):
        """
        Checks all bands in an image array to see if there is an entirely null value band. A tile (image array) with
        a single data band with all null values will be rejected later in the main code.

        :param multiband_img_arr: Multiband image array.
        :param nodata: No data value. Default set to -9999.

        :return: A boolean variable.
        """
        # initiating a variable is_img_null as False. If any of the band is entirely null, the function will
        # convert this variable to True. Otherwise, the function will return is_band_null as False.
        is_img_null = False

        # reading each band separately and checking if the entire band is null or not
        # if any of the band is entirely null, that tile won't be saved, as it has some missing attribute
        # Note that, an image where all bands have at least one valid pixel will pass this filter.
        for num_band in range(0, multiband_img_arr.shape[0]):
            single_arr = multiband_img_arr[num_band]

            if np.all(single_arr == nodata):
                is_img_null = True
            else:
                pass

        return is_img_null

    @staticmethod
    def count_perc_nodata(multiband_img_arr):
        """
        Calculates percent of nodata pixels in each band of an image array.

        :param multiband_img_arr: Multiband image array.

        :return: A list containing percent nodata pixel counts for all bands.
        """
        perc_counts_all_bands = []
        for num_band in range(0, multiband_img_arr.shape[0]):
            single_arr = multiband_img_arr[num_band]

            valid_pixels = np.count_nonzero(np.where(single_arr != -9999, 1, 0), keepdims=False)
            total_pixels = single_arr.shape[0] * single_arr.shape[1]
            nodata_count = total_pixels - valid_pixels

            perc_no_data = nodata_count * 100 / total_pixels

            perc_counts_all_bands.append(perc_no_data)

        return perc_counts_all_bands
