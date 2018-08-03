# adding header
print('\nWriting header into index.md ...')
with open('index.md', "r+") as f:
	old = f.read()
	f.seek(0)
	f.write("---\nlayout: projects\n\n---\n\n" + old)
print('Finished writing header\n')




# fixing latex equations
print('Fixing LaTeX equations ...')
import fileinput
with fileinput.FileInput('index.md', inplace=True) as file:
    for line in file:
        print(line.replace('\\[', '$$'), end='')
with fileinput.FileInput('index.md', inplace=True) as file:
    for line in file:
        print(line.replace('\\]', '$$'), end='')
        
        
# with fileinput.FileInput('index.md', inplace=True) as file:
#     for line in file:
#         print(line.replace('\\(', '$'), end='')

# with fileinput.FileInput('index.md', inplace=True) as file:
#     for line in file:
#         print(line.replace('\\)', '$'), end='')
  
print('Finished fixing LaTeX equations\n')




# removing extra gif in middle of md
print('Removing extra gif in middle of md ...')
with open('index.md', "r") as f:
	lines = f.readlines()

rm_line = '![](images/unnamed-chunk-9-1.gif)<!-- -->' + '\n'
with open('index.md', "w") as f:
	for line in lines:
		if line != rm_line:
			f.write(line)

print('Finished removing extra gif\n')



# converting tiff to png and then deleting tiff
print('Converting tiff to png, then deleting tiff...')
import subprocess
import os

# os.chdir(cwd+'/images')
subprocess.call('./images/convert.sh')

print('Finished converting.\n')