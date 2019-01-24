# Data aggregation Python 
# get JSON 
# get the tables
# append the URLs
# download all files to a folder
# OR edit data here and get rid of headers
import urllib.request
import requests
from bs4 import BeautifulSoup
import pandas as pd 

# Gets html of website
def get_html(base_url="http://www.fosr.org/data/"):
	base = base_url
	fp = urllib.request.urlopen(base)
	bytes_d = fp.read()

	return bytes_d

# Gets all extensions to download data
def get_extension_table(page):
	ext_table = []
	soup = BeautifulSoup(page)
	for link in soup.findAll('a'):
		ext = link.get('href')
		ext_table.append(ext)
	return ext_table[1:] # Get rid of irrelevant file

# Creates giant xls file with all data
def create_giant_xls(ext_table):
	base = "http://www.fosr.org"
	first_xls = pd.read_csv(base+ext_table[0], sep='\t')

	total_df = [first_xls]

	for ext in range(1, len(ext_table)):
		try:
				xls = pd.read_csv(base+ext_table[ext], sep='\t', encoding='cp1252')
				total_df.append(xls)
		except:
				print(base+ext_table[ext], "is not available right now")

	result = pd.concat(total_df, ignore_index=True)
	result.to_csv('water_quality_agg.csv', sep='\t', index=False, encoding='cp1252')
	# For first xls file, create new pandas, keep header
	# For each of rest add to pandas list
	# Take final pandas list, concatenate, download into csv/ xls if not possible
	return

page = get_html()
ext_page = get_extension_table(page)
create_giant_xls(ext_page)