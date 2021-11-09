<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta http-equiv="Content-Style-Type" content="text/css">
  <title>binseg_pytoch/README.md at master · saeedizadi/binseg_pytoch</title>
  <meta name="Description" content="Implementation of several state of the art methods for Binary Segmentation in PyTorch - binseg_pytoch/README.md at master · saeedizadi/binseg_pytoch">
  <meta name="Generator" content="Cocoa HTML Writer">
  <meta name="CocoaVersion" content="2113">
  <style type="text/css">
    p.p1 {margin: 0.0px 0.0px 0.0px 0.0px; font: 13.0px Courier; -webkit-text-stroke: #000000; min-height: 16.0px}
    p.p2 {margin: 0.0px 0.0px 0.0px 0.0px; font: 13.0px Courier; -webkit-text-stroke: #000000}
    span.s1 {font-kerning: none}
  </style>
</head>
<body>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1"># AECNN: Adversarial and Enhanced Convolutional Neural Networks</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1"># Abstract</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">Method for segmenting gastrointestinal polyps from colonscopy images uses an adversarial and enhanced convolutional neural networks (AECNN). As the number of training images is small, the core of AECNN relies on fine-tuning an existing deep CNN model (ResNet152). AECNN’s enhanced convolutions incorporate both dense upsampling, which learns to upsample the low-resolution feature maps into pixel-level segmentation masks, as well as hybrid dilation, which improves the dilated convolution by using different dilation rates for different layers. AECNN further boosts the performance of its segmenter by incorporating a discriminator competing with the segmenter, where both are trained through a generative adversarial network formulation.</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1"># Keywords</span></p>
<p class="p2"><span class="s1">Segmentation, Binary segmentation, Deep learning, polyp segmentation, Endoscopy analysis</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1"># Cite</span></p>
<p class="p2"><span class="s1">If you use our code, please cite our paper:<span class="Apple-converted-space"> </span></span></p>
<p class="p2"><span class="s1">[AECNN: Adversarial and Enhanced Convolutional Neural Networks](https://www2.cs.sfu.ca/~hamarneh/ecopy/caagv2021.pdf)</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">The corresponding bibtex entry is:</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">```</span></p>
<p class="p2"><span class="s1">@incollection{izadi2021aecnn,</span></p>
<p class="p2"><span class="s1"><span class="Apple-converted-space">  </span>title={AECNN: Adversarial and Enhanced Convolutional Neural Networks},</span></p>
<p class="p2"><span class="s1"><span class="Apple-converted-space">  </span>author={Izadi, Saeed and Hamarneh, Ghassan},</span></p>
<p class="p2"><span class="s1"><span class="Apple-converted-space">  </span>booktitle={Computer-Aided Analysis of Gastrointestinal Videos},</span></p>
<p class="p2"><span class="s1"><span class="Apple-converted-space">  </span>pages={59--62},</span></p>
<p class="p2"><span class="s1"><span class="Apple-converted-space">  </span>year={2021},</span></p>
<p class="p2"><span class="s1"><span class="Apple-converted-space">  </span>publisher={Springer}</span></p>
<p class="p2"><span class="s1">}</span></p>
<p class="p2"><span class="s1">```</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1"># The source code</span></p>
<p class="p2"><span class="s1">The implementation for several state of the art binary segmentation method in PyTorch. This repo was developed for participating in "[Endoscopic Vision Challenge<span class="Apple-converted-space">  </span>- Gastointetinal Image Analysis 2018](https://giana.grand-challenge.org)". And we won the 2nd place for polyp segmentation using DUCHDC model.<span class="Apple-converted-space"> </span></span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">Here are the implemented models:</span></p>
<p class="p2"><span class="s1">- [x] [FCN8]</span></p>
<p class="p2"><span class="s1">- [x] [FCN16]</span></p>
<p class="p2"><span class="s1">- [x] [FCN32]</span></p>
<p class="p2"><span class="s1">- [x] [SegNet]</span></p>
<p class="p2"><span class="s1">- [X] [PSPNet]</span></p>
<p class="p2"><span class="s1">- [x] [UNet]</span></p>
<p class="p2"><span class="s1">- [x] [Residual UNet]</span></p>
<p class="p2"><span class="s1">- [x] [DUC]</span></p>
<p class="p2"><span class="s1">- [x] [duchdc]</span></p>
<p class="p2"><span class="s1">- [x] [LinkNet]</span></p>
<p class="p2"><span class="s1">- [x] [FusionNet]</span></p>
<p class="p2"><span class="s1">- [x] [GCN]</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">## Usage</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">For quick hints about commands:</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">```</span></p>
<p class="p2"><span class="s1">python main.py -h</span></p>
<p class="p2"><span class="s1">```</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">Modify the condiguration in the `settings.py` file</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">### Training</span></p>
<p class="p1"><span class="s1"></span><br></p>
<p class="p2"><span class="s1">After customizing the `settings.py`, use the following command to start training</span></p>
<p class="p2"><span class="s1">```</span></p>
<p class="p2"><span class="s1">python main.py --cuda train</span></p>
<p class="p2"><span class="s1">```</span></p>
<p class="p2"><span class="s1">### Evaluation</span></p>
<p class="p2"><span class="s1">For evaluation, put all your test images in a folder and set path in the `settings.py`. Then run the following command:</span></p>
<p class="p2"><span class="s1">```</span></p>
<p class="p2"><span class="s1">python main.py --cuda eval</span></p>
<p class="p2"><span class="s1">```</span></p>
<p class="p2"><span class="s1">The results will be place in the `results` directory</span></p>
</body>
</html>
