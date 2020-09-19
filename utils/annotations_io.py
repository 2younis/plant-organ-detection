from lxml import etree as et


class Writer:

    def __init__(self, foldername, filename, imgSize, verified = False, localImgPath=None, databaseSrc='Unknown'):

        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = verified

    def addBndBox(self, xmin, ymin, xmax, ymax, name, score, difficult=0):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        bndbox['score'] = score
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = et.SubElement(top, 'object')
            name = et.SubElement(object_item, 'name')
            name.text = each_object['name']
            pose = et.SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = et.SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin']))== 1):
                truncated.text = "1" # max == height or min
            elif (int(float(each_object['xmax']))==int(float(self.imgSize[1]))) or (int(float(each_object['xmin']))== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            difficult = et.SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            score = et.SubElement(object_item, 'score')
            score.text = str(each_object['score'])

            bndbox = et.SubElement(object_item, 'bndbox')
            xmin = et.SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = et.SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = et.SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = et.SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def genXML(self):

        top = et.Element('annotation')

        if self.verified:
            top.set('verified', 'yes')

        folder = et.SubElement(top, 'folder')
        folder.text = self.foldername

        filename = et.SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = et.SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = et.SubElement(top, 'source')
        database = et.SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = et.SubElement(top, 'size')
        width = et.SubElement(size_part, 'width')
        height = et.SubElement(size_part, 'height')
        depth = et.SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = et.SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)

        et.indent(root, space="    ")

        prettifyResult = et.tostring(root, pretty_print=True).decode("utf-8")#.replace("  ".encode(), "\t".encode())

        with open(targetFile, 'w') as f:
            f.write(prettifyResult)


class Reader:

    def __init__(self, filepath):
        self.filepath = filepath
        self.shapes = []
        self.parseXML()

    def parseXML(self):

        with open(self.filepath) as f:
            xml_file = f.read()

        xmltree = et.fromstring(xml_file)
        
        self.filename = xmltree.find('filename').text
        self.folder = xmltree.find('folder').text
        self.path = xmltree.find('path').text

        size = xmltree.find('size')
        height = size.find('height').text
        width = size.find('width').text
        depth = size.find('depth').text

        self.imgSize = [height, width, depth]
        
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            label = object_iter.find('name').text
            difficult = object_iter.find('difficult').text
            truncated = object_iter.find('truncated').text
            
            #self.addShape(label, bndbox)
            bndbox = object_iter.find("bndbox")
            xmin = (float(bndbox.find('xmin').text))
            ymin = (float(bndbox.find('ymin').text))
            xmax = (float(bndbox.find('xmax').text))
            ymax = (float(bndbox.find('ymax').text))

            corners = [xmin, ymin, xmax, ymax]
            self.shapes.append((label, corners, difficult, truncated))
            
        
    def addShape(self, label, bndbox):
        xmin = (float(bndbox.find('xmin').text))
        ymin = (float(bndbox.find('ymin').text))
        xmax = (float(bndbox.find('xmax').text))
        ymax = (float(bndbox.find('ymax').text))

        corners = [xmin, ymin, xmax, ymax]

        self.shapes.append((label, corners, ))

