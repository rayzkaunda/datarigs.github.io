nb=$1

jupyter nbconvert $nb --to markdown
mv ${nb%.ipynb}.md ~/github/rayzkaunda/datarigs.github.io/_posts/
mv ${nb%.ipynb}_files ~/github/rayzkaunda/datarigs.github.io/images/
