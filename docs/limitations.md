# Limitations and Failure Modes

This page states known limitations directly, because several reviewer concerns concern underperformance scenarios.

## Severe out-of-support subdomains

Prototypes are fixed at inference. LSD-Seg cannot instantiate a new expert or prototype at test time. Out-of-support samples should be diagnosed through lower max prototype similarity, higher routing entropy, and higher uncertainty.

## Poor frozen encoder feature space

If the frozen ResNet, SAM, or SAM2 feature space fails to capture relevant medical texture, descriptor quality and routing can degrade. SAM2 is the paper-aligned path, but it still depends on the installed checkpoint and feature outputs.

## Prompt failure

Inaccurate prompts can affect the ROI descriptor and prompt prior. Prompt sensitivity analysis is provided to measure whether performance degrades gradually or fails abruptly under prompt corruption.

## Prototype fragmentation

Too low a novelty threshold or noisy descriptors can create fragmented prototypes. The merge threshold, minimum support, and merge interval mitigate this risk but do not eliminate it.

## 2D slice-wise modeling

The current framework is slice-wise 2D and does not explicitly enforce volumetric consistency. 3D or hybrid 2D/3D extensions are future work.

## Pseudo-subdomain circularity

Pseudo-clustered subdomains are useful diagnostics, but they are not a substitute for external metadata. When pseudo-subdomains are used, clustering must be fit on training data only and clearly reported.

