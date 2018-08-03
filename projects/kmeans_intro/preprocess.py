# adding header
print('\nWriting header into index.md ...')
with open('index.md', "r+") as f:
	old = f.read()
	f.seek(0)
	f.write("---\nlayout: projects\n\n---\n\n" + old)
print('Finished writing header\n\n')


# removing extra gif in middle of md
print('\nRemoving extra gif in middle of md ...')
with open('index.md', "r") as f:
	lines = f.readlines()

rm_line = '![](images/unnamed-chunk-10-1.gif)' + '\n'
with open('index.md', "w") as f:
	for line in lines:
		if line != rm_line:
			f.write(line)

print('\nFinished removing extra gif\n\n')



# converting tiff to png and then deleting tiff
print('Converting tiff to png, then deleting tiff...')
import subprocess
import os

# os.chdir(cwd+'/images')
subprocess.call('./images/convert.sh')

print('Finished converting.\n')