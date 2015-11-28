# Automatically Generate Tags
## Structure

#### data
 + data folder contains the raw data of this project, namely anime.txt.

#### dump
 + dump folder contains the results in the form of pickle.

#### fig
 + fig folder contains all the figures drawn to visualize the results.

#### reports
 + reports folder contains proposal, midway milestone, poster, and final report of the project.

#### src
 + src folder contains all the source code.

## Instruction of running the code
 + Run GeTagAnime.py to get all the classification result and F scores, the results will be dumped into pickle files in dump folder.
 + Run draw.py to generate figures from the dumped results. Figures will be saved in fig folder.

## Libraries needed before run
 + pdb
 + numpy
 + matplotlib
 + scikit-learn : conda install scikit-learn in the shell
 + pickle
 + nltk :
   ```
   conda install nltk
   
   python
   
   >>> import nltk
   
   >>> nltk.download()
   ````
   Then choose stopwords to install.
   
+ have fun!!
