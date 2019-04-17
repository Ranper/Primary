1	baike_data_shawn_94873_main.txt

Each column means an entity:
id	existing time(day)	edit times	view times	inner links number(including subview)	links number	inner links number	text length	main content length	title length
每列分别代表词条的：
id 时间T 编辑次数 浏览次数 链接到百科词条链接数 链接数 只链接到ids的链接数 所有文本长度 正文文本长度 标题长度
注1：每个词条对应的时间T = （收集时间-创建时间）/30   文件中的时间单位是天，实际训练中单位时间取的是月，所以时间需要除以30
注2：id对应url e.g. url为http://baike.baidu.com/view/1.htm 对应的id为1
注3：所有的文件都是按id排序。
注4：链接到百科词条链接数 统计的是http://baike.baidu.com/view/...和http://baike.baidu.com/subview/...的两者的链接数
	 链接数 统计的是词条页面内所有超链接数
	 链接到ids的链接数 统计的是http://baike.baidu.com/view/...的链接数

2	baike_data_shawn_94873_newY.txt

Edit times in the last one month
new的更新频率  统计2016年12月的更新次数
每一行对应一个词条。每一行包含 id	更新次数

3	baike_data_shawn_94873_link_ids_exisit.txt
links data
存储的是一个词条页面中链接词条    覆盖率42%
每一行对应一个词条。每一行包含 本词条的id + 该词条链接词条的id

4	baike_data_shawn_94873_content.txt
content data
每一行对应一个词条。每一行包含 id:::content

5	baike_data_shawn_94873_hist_noe.txt
The edit time of each update for each entity
每一行对应一个词条。每一行包含 id history page1 的更新时间
需要由此得到新的更新频率   而不是由总编辑次数/总时间T得到
