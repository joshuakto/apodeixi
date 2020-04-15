class bat:
	def __init__(self, bid = 0, Contents={}):
		self.bid = bid
		self.Contents = {}
	def add_content(self, info):
		temp_content = Content()
		temp_content.load(info)
		temp_dict = {info['content_id']:temp_content}
		self.Contents = {**self.Contents, **temp_dict}

class Content:
	def __init__(self, content_id=None, 
		     sen1=None, sen2=None):
		self.content_id = content_id
		self.sen1 = sen1
		self.sen2 = sen2

	def __str__(self):
		return ''.join(('Content: {self.content_id}\n'.format(self=self),
			'Sentence 1: {self.sen1}\n'.format(self=self),
			'Sentence 2: {self.sen2}'.format(self=self)))
		
	def load(self, info):
		self.content_id = info['content_id']
		self.sen1 = info['sen1']
		self.sen2 = info['sen2']

	def testpd(self)
		print(pd)
