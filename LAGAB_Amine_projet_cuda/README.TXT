vous retrouverez 3 dossiers : 
effects => dossier contenant tout les .cu des effets implémenté .
example => dossier contenant les .cu de l'exemple choisi pour le profiling

plateforme_compil => le projet visual basic studio qui malheureusement ne permet pas l'exe de plusieurs fichiers .cu à la fois 
                    le sous dossier plateforme_compil\Convolutions contient les images , la library Handle_errors et le .cu à compiler
                    lors de la compil.
                     les .cu ne peuvent pas s'exe simultanément : erreur de build .
                    pour exécuter un .cu se trouvant dans le dossier "effects" , je copie le code du programme et je le colle dans 
                    Conv1.cu se trouvant ici "plateforme_compil\Convolution" (je n'ai pas trouvé d'autres solutions).

pour le profiling , j'ai utiliser le chrono pour cpu et les events pour gpu : 
 je déroule mon programme 1000 fois , à chaque fois je récupère le temps d'exe que je mets dans un fichier .csv , à la fin j'ai 1000 lignes
de temps d'exe que je traite sur R studio afin de générer la moyenne , le min , le max . j'ai opté pour cette démarche pour avoir un maximum
de cohérence .

