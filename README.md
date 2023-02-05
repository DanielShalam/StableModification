# StableModification
 Modify images with stable diffusion
 
## How to run:
 ### Configuration:
  The files under configs dir are an example of how the configuration is built for each application.
  
  You can add experiment by insert a new id ("0", "1", ...) with the new parameters.
  
  ```
  prompt_init: The source prompt from which the image should be generated/inverted.
  
  prompt_new: The new prompt with your modifications.
  
  img_path: If it given, it will triger null inversion of the image. If you want to use it without, leave it blank ("").
  
  steps: How many denoising steps.
  
  scale: The uncoditional guidance scale.
  
  self_replace_steps: Controlling the replacement of the self-attention maps. (Should be <= 1, default is 0.4).
  
  cross_replace_steps: Controlling the replacement of the cross-attention maps. (Should be <= 1, default is 0.8).
  
  localize:
  
     apply: If to perform localization (edit only areas with this context).
     
     words: Words to localize e.g. ["trees", "trees"]
     
  reweight:
  
     apply: If to perform reweightning for specific words.
     
     words: List of words.
     
     weights: New weights.
     
  refine:
  
     apply: If to perform refinement (usually good when adding words the the source text).
     
  replace:
  
     apply: If to replace specific words.
     
  ```
Note that only one from refine/replace can be activated.

 ### commands:
 Run:
 ```
 python prompt2prompt.py --config <path to config file> --outdir <outputs directory>
 ```
