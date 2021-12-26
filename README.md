# Nested UNet (UNet++) for Building Detection

## Demo

| <img src="img/val_img.png" alt="val_img" width="400px"> | <img src="img/val_label.png" alt="val_label" width="400px"> | <img src="img/output.png" alt="output" width="400px"> | <img src="img/CRF.png" alt="CRF" width="400px"> | <img src="img/denoised.png" alt="denoised" width="400px"> |
|:-------------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
| Input Image | Ground Truth | UNet++ Output | Post-prosessed by Fully Connected CRF |  Post-prosessed by Fully Connected CRF and Denoising |

<table>
    <thead>
        <tr>
            <th align="center" style="width:20%"><a target="_blank" rel="noopener noreferrer" href="/hashi0203/Nested-UNet-Building-Detection/blob/main/img/val_img.png"><img src="img/val_img.png" alt="val_img" width="400px" style="max-width: 100%;"></a></th>
            <th align="center" style="width:20%"><a target="_blank" rel="noopener noreferrer" href="/hashi0203/Nested-UNet-Building-Detection/blob/main/img/val_label.png"><img src="img/val_label.png" alt="val_label" width="400px" style="max-width: 100%;"></a></th>
            <th align="center" style="width:20%"><a target="_blank" rel="noopener noreferrer" href="/hashi0203/Nested-UNet-Building-Detection/blob/main/img/output.png"><img src="img/output.png" alt="output" width="400px" style="max-width: 100%;"></a></th>
            <th align="center" style="width:20%"><a target="_blank" rel="noopener noreferrer" href="/hashi0203/Nested-UNet-Building-Detection/blob/main/img/CRF.png"><img src="img/CRF.png" alt="CRF" width="400px" style="max-width: 100%;"></a></th>
            <th align="center"><a target="_blank" rel="noopener noreferrer" href="/hashi0203/Nested-UNet-Building-Detection/blob/main/img/denoised.png"><img src="img/denoised.png" alt="denoised" width="400px" style="max-width: 100%;"></a></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Input Image</td>
            <td align="center">Ground Truth</td>
            <td align="center">UNet++ Output</td>
            <td align="center">Post-prosessed by Fully Connected CRF</td>
            <td align="center">Post-prosessed by Fully Connected CRF and Denoising</td>
        </tr>
    </tbody>
</table>