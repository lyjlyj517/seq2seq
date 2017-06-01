#! python coding:utf-8

import sys
import json

reload(sys)
sys.setdefaultencoding('utf-8')
argvs = sys.argv
argc = len(argvs)

if argc < 2:
	print "usage:python argvs[0] filename"
else:

	input = argvs[1]
	f=open(input, "r")
	json_data = json.load(f)

	print "dialogue-id : " + json_data["dialogue-id"]
	print "speaker-id : " + json_data["speaker-id"]
	print "group-id : " + json_data["group-id"]
		
	for turn in json_data["turns"]:
		s = turn["speaker"] + ":" + turn["utterance"]
		a = " "
		for annotate in turn["annotations"]:
			a = a + annotate["breakdown"] + " "
		s = s + a
		print s

