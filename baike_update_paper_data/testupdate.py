# coding=utf-8


import ljqpy, time, re, random
from datetime import datetime, timedelta
import urllib.parse
from bs4 import BeautifulSoup
from pymongo import MongoClient
client = MongoClient('mongodb://:27017/')
db = client.cndbpedia
time.clock()
import threading
from dbtools import MatchExisted

import jieba

def Logging(sr):
	print(str(sr)[:100])
	
	
def CheckExisted(ent):
	zz = db.entities.find_one({'_id': ent})
	if zz is None: return False
	return True

def MakeALabel(z):
	z = str(z).strip().replace('\xa0', '').replace('&nbsp;', ' ')
	z = re.sub('[\r\n\t]', '', z)
	z = re.sub('<sup>.+?</sup>', '', z)
	z = re.sub('<a .+?>', '*a*', z).replace('</a>', '*/a*')
	z = re.sub('<br/?>', '\n', z)
	z = re.sub('<.+?>', '', z)
	z = z.replace('*a*', '<a>').replace('*/a*', '</a>').strip()
	z = z.replace('<a></a>', '').strip()
	return z

def DoInsert(x): print(x)



def Extract(page, wd='', update=True):
	global soup
	soup = BeautifulSoup(page, 'html.parser')
	try:
		if not soup.title.text.endswith('_百度百科'): return []
		name = soup.title.text.replace('_百度百科', '').lower()
		if name == '': return []
		polys = []
		poly = soup.find('ul', 'polysemantList-wrapper')
		if poly is not None:
			for item in poly.find_all('li', 'item'):
				if item.a is not None:
					polys.append(item.a.get('href'))
		existed = CheckExisted(name)
		if not update and existed: return polys
		infobox = soup.find('div', 'basic-info')
		kvs = []
		if infobox is not None:
			ks = infobox.find_all('dt')
			vs = infobox.find_all('dd')
			for k,v in zip(ks, vs):
				k = k.text.strip().replace('\xa0', '')
				v = MakeALabel(v)
				kvs.append( (k, v) )
		summ = soup.find('div', 'lemma-summary')
		paras = [MakeALabel(x) for x in summ.find_all('div', 'para')]
		paras = [x for x in paras if x != '']
		if len(paras) > 0:
			desc = '\n'.join(paras)
			kvs.append( ('DESC', desc) )
		print(name, len(kvs), 'triples')
		if len(kvs) == 0: return polys
		return kvs
		for x in kvs: ScanLinks(x[-1])
		if wd != '' and wd != name: DoInsert(['ment2ent', wd, name])
		DoInsert( ['triples', name, kvs] )
		DoInsert( ['htmls', name, wd, page])
		DoInsert( ['entities', name])
		return polys
	except Exception as e:
		print(e)
	return []
		
def ScanLinks(sr):
	global queue, overset
	zs = {z.lower() for z in re.findall('<a>(.+?)</a>', sr)}
	for z in zs:
		if not z in overset:
			if MatchExisted(z) != '': continue 
			overset.add(z)
			queue.append(z)

def SearchBaike(wd):
	global page
	url = 'http://baike.baidu.com/item/%s' % wd
	page = ljqpy.GetPage(url)
	polys = Extract(page, wd)
	for up in polys:
		Extract(ljqpy.GetPage(urllib.parse.urljoin(url, up)), wd)
		time.sleep(2)


overset = set()
queue = []
waiting = 12

name = 'UPDATE'
desc = 'by LJQ'
examples = []
port = 59281


def Background():
	global queue, waiting
	while True:
		time.sleep(3)
		waiting -= 3
		if waiting < 0:
			z = ''
			print(queue)
			if len(queue) > 0: z = queue.pop()
			if len(z) > 0: SearchBaike(z)
			waiting = 121

#threading._start_new_thread(Background, ())

def Run(param):
	global overset, queue, waiting
	z = param.strip().lower()
	if not z in overset:
		overset.add(z)
		queue.append(z)
		waiting = 0
	return '500 SERVER ERROR'

def FindNewWords():
	import hotwords as hw
	funcs = [hw.GetSougou, hw.GetTieba]
	wds = []
	for func in funcs:
		try:
			wds += func()
		except Exception as e:
			print(func, e)
	wds = hw.Filter(list(set(wds)))
	print(wds, len(wds))
	return wds


def FirstStep():
	wds = FindNewWords()
	with open('tests/first.txt', 'w', encoding='utf-8') as fout:
		for wd in wds:
			print(wd)
			url = 'http://baike.baidu.com/item/%s' % wd
			page = ljqpy.GetPage(url)
			soup = BeautifulSoup(page, 'html.parser')
			if not soup.title.text.endswith('_百度百科'): continue
			name = soup.title.text.replace('_百度百科', '').lower()
			page = re.sub('[\r\t\n]', ' ', page)
			page = re.sub('[ ]+', ' ', page)
			ljqpy.WriteLine(fout, [wd, name, page])


def SecondStep():
	dat = ljqpy.LoadCSV('tests/first.txt')
	with open('tests/second.txt', 'w', encoding='utf-8') as fout:
		for zz in dat:
			kvs = Extract(zz[2], zz[0])
			zs = set()
			for k, sr in kvs: zs = zs.union(z.lower() for z in re.findall('<a>(.+?)</a>', sr))
			for z in zs:
				ljqpy.WriteLine(fout, [zz[1], z])

def ThirdStep():
	wds = {x[1] for x in ljqpy.LoadCSV('tests/second.txt')}
	with open('tests/third.txt', 'w', encoding='utf-8') as fout:
		for wd in wds:
			print(wd)
			url = 'http://baike.baidu.com/item/%s' % wd
			page = ljqpy.GetPage(url)
			soup = BeautifulSoup(page, 'html.parser')
			if not soup.title.text.endswith('_百度百科'): continue
			name = soup.title.text.replace('_百度百科', '').lower()
			page = re.sub('[\r\t\n]', ' ', page)
			page = re.sub('[ ]+', ' ', page)
			ljqpy.WriteLine(fout, [wd, name, page])

def FourthStep():
	patt = '<li>最近更新：<span class="j-modified-time" style="display:none">(.+?)</span>'
	dat = ljqpy.LoadCSV('tests/third.txt')
	#dat = [['1', '1', open('z:/aa.txt', encoding='utf-8').read()]]
	dtt = ljqpy.LoadCSV('tests/expand_hist_editfreq1.txt')
	dtt = {x[0]:x for x in dtt}
	with open('tests/fourth.txt', 'w', encoding='utf-8') as fout:
		for zz in dat:
			page = zz[2]
			soup = BeautifulSoup(page, 'html.parser')
			tt = ljqpy.RM(patt, page)
			if tt == '': continue
			if not zz[0] in dtt: continue
			vdays = (datetime.strptime('2016-12-31', '%Y-%m-%d') - datetime.strptime(dtt[zz[0]][3], '%Y-%m-%d %H:%M:%S')).total_seconds() // 86400
			fv = [vdays]
			fv.append(int(ljqpy.RM('<li>编辑次数：([0-9]+)次', page)))
			lemmapv = ljqpy.RM('newLemmaIdEnc:"(.+?)"', page)
			lemmapg = ljqpy.GetPage('http://baike.baidu.com/api/lemmapv?id=%s' % lemmapv)
			fv.append(int(ljqpy.RM('([0-9]+)', lemmapg)))
			fv.append(page.count('href="/vi') + page.count('href="/sub'))
			fv.append(page.count('href="'))
			paras = [re.sub('<.+?>', '', MakeALabel(x)) for x in soup.find_all('div', 'para')]
			paras = [x for x in paras if x != '']
			fv.append(sum(len(x) for x in paras))
			summ = soup.find('div', 'lemma-summary')
			paras = [re.sub('<.+?>', '', MakeALabel(x)) for x in summ.find_all('div', 'para')]
			paras = [x for x in paras if x != '']
			fv.append(sum(len(x) for x in paras))
			fv.append(len(zz[0]))
			print(zz[0], fv)
			ljqpy.WriteLine(fout, [zz[0], tt] + fv)

def GoodDate(dd):
	return datetime.strptime(dd, '%Y-%m-%d') > datetime.strptime('2016-12-16', '%Y-%m-%d')

import math
def TestPred(fn, di, vi, ran=False):
	global prcret
	data = ljqpy.LoadCSV(fn)
	if ran:
		random.shuffle(data)
	good, tot = 0, 0
	for dd in data: 
		if GoodDate(dd[di]): tot += 1
	oo = True
	prcret = []
	for kk, dd in enumerate(data):
		if kk > 0:
			prec, reca = good / kk, good / tot
			prcret.append((prec, reca))
		if kk % 20 == 0 and kk > 0:
			if kk < 100 or kk % 50 == 0: 
				f1 = 2 * prec * reca / (prec + reca)
				print('Prec@%d: %d/%d %.3f  Reca@%d: %d/%d %.3f  f1: %.3f' % (kk, good, kk, prec, kk, good, tot, reca, f1))
		if GoodDate(dd[di]): good += 1
		#if kk > 100: break
	auc = 0
	for ii in range(1, len(prcret)):
		auc += (prcret[ii-1][0] + prcret[ii][0]) * (prcret[ii][1] - prcret[ii-1][1]) * 0.5
	print('auc = %.5f' % auc)

	rdata = [1 if GoodDate(dd[di]) else 0 for dd in data]
	MAP, tp = 0, 0
	for k, v in enumerate(rdata):
		if v != 1: continue
		tp += 1
		MAP += tp / (k+1)
	MAP /= tp
	print('MAP = %.4f, tp = %d' % (MAP, tp))
	DCG, IDCG = 0, 0
	for k, v in enumerate(rdata):
		DCG += v / math.log2(k+2)
	for k, v in enumerate(sorted(rdata, reverse=True)):
		IDCG += v / math.log2(k+2)
	print('DCG = %.4f, IDCG = %.4f, nDCG = %.4f' % (DCG, IDCG, DCG / IDCG))



def TestRandomSeed():
	zz = list(db.entities.find({}).skip(random.randint(1, 99999)).limit(100))
	zz += list(db.entities.find({}).skip(random.randint(1, 999999)).limit(100))
	zz += list(db.entities.find({}).skip(random.randint(1, 9999999)).limit(100))
	zz += list(db.entities.find({}).skip(random.randint(1, 9999999)).limit(100))
	wds = [x['_id'] for x in random.sample(zz, 130)]
	print(wds)
	with open('tests/randomseeds.txt', 'w', encoding='utf-8') as fout:
		for wd in wds:
			print(wd)
			url = 'http://baike.baidu.com/item/%s' % wd
			page = ljqpy.GetPage(url)
			soup = BeautifulSoup(page, 'html.parser')
			if not soup.title.text.endswith('_百度百科'): continue
			name = soup.title.text.replace('_百度百科', '').lower()
			page = re.sub('[\r\t\n]', ' ', page)
			page = re.sub('[ ]+', ' ', page)
			ljqpy.WriteLine(fout, [wd, name, page])

def TestRandomSeedTime(fn):
	data = ljqpy.LoadCSV(fn)[:87]
	patt = '<li>最近更新：<span class="j-modified-time" style="display:none">(.+?)</span>'
	with open('%s_lastupd.txt' % fn[:-4], 'w', encoding='utf-8') as fout:
		for zz in data:
			ljqpy.WriteLine(fout, [zz[0], zz[1], ljqpy.RM(patt, zz[2])])

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')
def CheckUpdatingDist():
	fns = ['tests/randomseeds_lastupd.txt', 'tests/first_lastupd.txt']
	plt.rc('xtick', labelsize=20)
	plt.rc('ytick', labelsize=20)
	plt.figure()
	for fn in fns:
		dd = ljqpy.LoadCSV(fn)
		dd = [x for x in dd if x[2] != '']
		ds = [(datetime.strptime('2017-01-19', '%Y-%m-%d')-datetime.strptime(x[2], '%Y-%m-%d')).total_seconds() // 86400 for x in dd]
		print(fn)
		print(ds)
		ds = np.array(ds)
		print((ds<31).sum())
		dx = range(1, 400)
		plt.plot(dx, [np.sum(ds<=x) / len(ds) for x in dx])
	plt.xlabel('X', fontsize=20)
	plt.ylabel('Pr(d<X)', fontsize=20)
	plt.legend(['Random', 'Hot seeds'], loc=4, fontsize=20)
	plt.show()

def GetHistoryFreq():
	dd = ljqpy.LoadCSVg('tests/third.txt')
	patt = '"(/historylist/.+?)"'
	pattlastu = '<li>最近更新：<span class="j-modified-time" style="display:none">(.+?)</span>'
	outfn = 'tests/expand_hist_editfreq1.txt'
	odate = datetime.strptime('2016-12-01', '%Y-%m-%d')
	with open(outfn, 'w', encoding='utf-8') as fout:
		for z in dd:
			try:
				print(z[0])
				url = 'https://baike.baidu.com' + ljqpy.RM(patt, z[2])
				print(url)
				pg = ljqpy.GetPage(url)
				totaledit = int(ljqpy.RM('共被编辑([0-9]+)次', pg))
				print(totaledit)
				pgnum = (totaledit + 24) // 25
				tk = ljqpy.RM('tk.+?=.+?"(.+?)";', pg)
				lemmaId = ljqpy.RM('lemmaId.+?=.+?([0-9]+?);', pg)
				for zz in re.findall('<td>([0-9-]+) .+?</td>', pg):
					if datetime.strptime(zz, '%Y-%m-%d') > odate: totaledit -= 1
				#print(tk, lemmaId)
				uurl = 'https://baike.baidu.com/api/wikiui/gethistorylist?tk=%s&lemmaId=%s&from=%d&count=1&size=25' % (tk, lemmaId, pgnum)
				pg = ljqpy.GetPage(uurl)
				sdate = re.findall('"createTime":([0-9]+),', pg)[-1]
				sdate = datetime.fromtimestamp(int(sdate))
				print(totaledit, sdate.strftime('%Y-%m-%d'))
				weekfreq = totaledit / ((odate - sdate).total_seconds() // 86400 / 7)
				ljqpy.WriteLine(fout, [z[0], z[1], totaledit, sdate, '%.3f' % weekfreq, ljqpy.RM(pattlastu, z[2])])
				time.sleep(2)
			except: pass
	data = ljqpy.LoadCSV(outfn)
	data.sort(key=lambda d:-float(d[4]))
	ljqpy.SaveCSV(data, outfn)


plt.figure()
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)

plt.xlabel('Recall', fontsize=25)
plt.ylabel('Precision', fontsize=25)


if __name__ == '__main__':
	#Run('薛之谦')
	#while True: FindNew()
	#ThirdStep()
	#FourthStep()
	#TestPred()
	#TestRandomSeed()
	#TestRandomSeedTime('tests/randomseeds.txt')
	#TestRandomSeedTime('tests/first.txt')
	CheckUpdatingDist()
	#GetHistoryFreq()
	random.seed(1313)
	TestPred('tests/ret.txt', 1, 2, True)
	plt.plot([y for x,y in prcret], [x for x,y in prcret], color='black')
	TestPred('tests/expand_hist_editfreq1.txt', -1, 4)
	plt.plot([y for x,y in prcret], [x for x,y in prcret], color='b')
	TestPred('tests/ret_linear.txt', 1, 2)
	plt.plot([y for x,y in prcret], [x for x,y in prcret], color='g')
	TestPred('tests/ret.txt', 1, 2)
	plt.plot([y for x,y in prcret], [x for x,y in prcret], color='r')
	plt.legend(['Random', 'Baseline', 'Linear', 'Random forest'], loc=4, fontsize=20)
	plt.show()
	print('completed %.3f' % time.clock())


