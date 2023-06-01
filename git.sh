#create a new git
git init
git add *
git commit -m "Start new repository"
git branch -M main
git remote add origin https://github.com/kieucq/da_3dvar_L40.git
git push -u origin main

# adding files/folders
git add letkf/*.ctl
git commit -m "Adding Grads ctl files of letkf"
git push -u origin main

# update all changes
git status -uno # opt -uno will not list all untracked files
git add -u
git commit -m "Adding all modified files only"
git push -u origin main

# remove files/folder
git rm letkf/*.dat
git commit -m "Remove all dat files"
git push -u origin main

#Personal access token
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
