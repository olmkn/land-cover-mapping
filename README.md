# Pixel-based Remote Sensing Image Classification

*This project focuses on supervised classification of Sentinel-2 satellite imagery to map land cover types over an area in the Skadovsk rayon, Kherson oblast. By leveraging feature engineering and machine learning techniques, this project achieved a peak accuracy of 94% using ANN in identifying diverse land cover classes.*

---

## Tools and Libraries

- SAGA GIS
- QGIS
- R (V.4.1.2)
	- `caret` (V.6.0-88)
	- `randomForest` (V.4.6-14)
	- `nnet` (V.7.3-16)
	- `parallel` (V.4.1.2) 
	- `doParallel` (V.1.0.16)

## Data source

- **Source**: Sentinel-2 (Copernicus program, European Space Agency)
- **Region**: Skadovsk rayon, Kherson oblast.
- **Imagery**: Sentinel-2B (4 May, 2 August 2021) and Sentinel-2A (26 October 2021).
- **Resolution**: 10 m/pixel (resampled for uniformity).
- **Features**: Spectral bands (10 channels) and indices (e.g., NDVI).

## Land cover classes

| Class number |           Class name            | Label (full) | Label (abbrv.) | Reference polygons | Training sample <br>(# of pixels) | Test sample <br>(# of pixels) | Total |
| :----------: | :-----------------------------: | :----------: | :------------: | :----------------: | :-------------------------------: | :---------------------------: | :---: |
|      1       | Agricultural land (arable land) | Agriculture  |       AG       |         46         |               2261                |              969              | 3230  |
|      2       |          Alder forests          |    Alnus     |       AL       |         24         |               2649                |             1134              | 3783  |
|      3       |       Artificial surfaces       |   BuiltUp    |       BU       |         72         |               2309                |              989              | 3298  |
|      4       |            Reed beds            |  Phragmites  |       Ph       |         24         |               2224                |              953              | 3177  |
|      5       |          Pine forests           |    Pinus     |       P        |         43         |               2401                |             1029              | 3430  |
|      6       |         Robinia forests         |   Robinia    |       R        |         43         |               2527                |             1083              | 3610  |
|      7       |           Bare sands            |    Sands     |       S        |         45         |               1622                |              695              | 2317  |
|      8       |       Dense sandy steppe        | SandsSteppe  |      SSt       |         31         |               1461                |              625              | 2086  |
|      9       | Sandy steppe of medium density  | SandsSteppe1 |      SSt1      |         17         |               1959                |              839              | 2798  |
|      10      |       Sparse sandy steppe       | SandsSteppe2 |      SSt2      |         49         |               1770                |              758              | 2528  |
|      11      |              Water              |    Water     |       W        |         83         |               2476                |             1060              | 3536  |
|              |                                 |              |                |        477         |               23659               |             10134             | 33793 |

