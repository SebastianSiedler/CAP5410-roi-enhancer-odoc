The `glaucoma-datasets` folder consists of retinal fundus images from three sources:

- G1020 (1020 images)
- ORIGA (650 images)
- REFUGE (1200 images)

The dataset is organized into the following structure:

- glaucoma-datasets/
  - G1020/
    - Images_Cropped/img/
    - Masks_Cropped/img/
  - ORIGA/
    - Images_Cropped/img/
    - Masks_Cropped/img/
  - REFUGE/
    - test/Images_Cropped/
    - test/Masks_Cropped/
    - train/Images_Cropped/
    - train/Masks_Cropped/
    - val/Images_Cropped/
    - val/Masks_Cropped/

The `Masks_Cropped` folder contains png files with values from 0 to 2, where:

- 0: background
- 1: optic disc
- 2: optic cup
