import os
import sys
from frechet_audio_distance import FrechetAudioDistance

frechet = FrechetAudioDistance(
    use_pca=False, 
    use_activation=False,
    verbose=True
)
#path=sys.argv[1]
bg=sys.argv[1]
test=sys.argv[2]
#print(path)
#print(bg)
#print(test)
print("Computing FAD of: ",test, " using background: ",bg)
fad_score = frechet.score(bg,test)
print("Result FAD: ",fad_score)

