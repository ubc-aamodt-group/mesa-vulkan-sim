from enum import Enum, auto, EnumMeta
import re

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True

class InstructionClass(Enum):
    VariableDeclaration = auto()
    EntryPoint = auto()
    Functional = auto()
    Empty = auto()
    UNKNOWN = auto()

class PTXLine:
    def __init__(self, fullLine) -> None:
        if fullLine.count(';') > 1:
            print('ERROR: you can write only one instruction per line')
            exit(-1)
        
        self.fullLine = fullLine
        if '//' in fullLine:
            self.command, self.comment = fullLine.split("//", 1)
        else:
            self.command = fullLine
            self.comment = ''
        self.leadingWhiteSpace = re.match(r"\s*", self.command).group()
        self.command = self.command.strip()
        self.instructionClass = PTXLine.getInstructionClass(fullLine)
        self.condition = ''
    
    def buildString(self):
        self.fullLine = self.condition + self.leadingWhiteSpace + self.command
        if len(self.comment) > 0:
            self.fullLine += '; //' + self.comment
        else:
            self.fullLine += ';\n'
    
    def addComment(self, comment):
        self.comment += '//' + comment + '\n'
        self.buildString()

    @staticmethod
    def getInstructionClass(line):
        if len(line) == 0 or line.isspace():
            return InstructionClass.Empty
        
        #print("-%s-%s" % (line, line.isspace()))
        firstWord = line.split(None, 1)[0]
        if firstWord == '.entry':
            return InstructionClass.EntryPoint
        
        if firstWord == '.reg' or firstWord == '.local':
            return InstructionClass.VariableDeclaration
        
        if firstWord in FunctionalType or firstWord.split('.')[0] in FunctionalType:
            return InstructionClass.Functional
        
        if firstWord[0] == '.':
            return InstructionClass.UNKNOWN
        
        if firstWord[0:2] == '//':
            return InstructionClass.Empty
        
        return InstructionClass.Functional
    
    @staticmethod
    def createNewLine(line):
        lineClass = PTXLine.getInstructionClass(line)
        if lineClass == InstructionClass.VariableDeclaration:
            return PTXDecleration(line)
        elif lineClass == InstructionClass.EntryPoint:
            return PTXEntryPoint(line)
        elif lineClass == InstructionClass.Functional:
            return PTXFunctionalLine(line)
        else:
            return PTXLine(line)


class DeclarationType(Enum):
    Register = '.reg'
    Local = '.local'
    Other = auto()


class PTXDecleration (PTXLine):
    def __init__(self, fullLine = '') -> None:
        super().__init__(fullLine)
        if self.command != '':
            self.parse(self.command)
    
    def parse(self, command):
        print('###')
        print(command)
        print('###')
        self.command = command
        firstWord = command.split(None, 1)[0]
        if firstWord == '.reg':
            self.declarationType = DeclarationType.Register
        elif firstWord == '.local':
            self.declarationType = DeclarationType.Local
        else:
            raise NotImplementedError
        
        args = command.split()
        if '.v' in args[1]:
            self.vector = args[1]
            index = 2
        else:
            self.vector = ''
            index = 1
        
        self.variableType = args[index]
        self.variableName = args[index + 1]
        if self.variableName[-1] == ';':
            self.variableName = self.variableName[:-1]
        
        if index + 2 < len(args):
            args = args[index + 2:]
        else:
            args = None
    
    def buildString(self, declarationType, vector, variableType, variableName):
        assert declarationType != DeclarationType.Other

        self.declarationType = declarationType
        self.vector = vector
        self.variableType = variableType
        self.variableName = variableName

        if self.vector == None:
            vector = ''
        else:
            vector = ' ' + self.vector
        self.command = '%s%s %s %s' % (declarationType.value, vector, variableType, variableName)
        super().buildString()
    
    def isVector(self):
        if self.vector is None or self.vector == '':
            return False
        else:
            return True
    
    def vectorSize(self):
        assert self.isVector()
        assert self.vector[0:2] == '.v'
        return int(self.vector[2:])
    
    def bitCount(self):
        return int(self.variableType[2:])


class FunctionalType(Enum, metaclass=MetaEnum):
    load_ray_launch_id = 'load_ray_launch_id'
    load_ray_launch_size = 'load_ray_launch_size'
    vulkan_resource_index = 'vulkan_resource_index'
    load_vulkan_descriptor = 'load_vulkan_descriptor'
    deref_cast = 'deref_cast'
    deref_struct = 'deref_struct'
    deref_array = 'deref_array'
    load_deref = 'load_deref'
    store_deref = 'store_deref'
    trace_ray = 'trace_ray'
    call_miss_shader = 'call_miss_shader'
    call_closest_hit_shader = 'call_closest_hit_shader'
    alloca = 'alloca'
    decl_var = 'decl_var'
    deref_var = 'deref_var'
    mov = 'mov'
    image_deref_store = 'image_deref_store'
    exit = 'exit'
    ret = 'ret'
    phi = 'phi'
    Other = auto()

class PTXFunctionalLine (PTXLine): # come up with a better name. I mean a line that does sth like mov (eg it's not decleration)
    def __init__(self, fullLine = '') -> None:
        super().__init__(fullLine)
        if self.command != '':
            self.parse(self.command)
    
    def parse(self, command):
        self.command = command

        firstWord = command.split(None, 1)[0]
        if firstWord[-1] == ';':
            firstWord = firstWord[:-1]

        if firstWord in FunctionalType:
            self.functionalType = FunctionalType[firstWord]
            self.fullFunction = self.functionalType
        elif firstWord.split('.')[0] in FunctionalType:
            dotSplit = firstWord.split('.')
            self.functionalType = FunctionalType[dotSplit[0]]
            self.fullFunction = firstWord
            self.vector = None
            for specifier in dotSplit[1:]:
                if specifier[0] == 'v':
                    self.vector = '.' + specifier
                else:
                    self.variableType = '.' + specifier

        else:
            self.functionalType = FunctionalType.Other
            self.fullFunction = firstWord
        

        if len(command.split(None, 1)) > 1:
            args = command.split(None, 1)[1]
            if args[-1] == ';':
                args = args[:-1]
            elif args[-1][-1] == ';':
                args[-1] = args[-1][:-1]
            args = args.split(',')
            self.args = [arg.strip() for arg in args]
        else:
            self.args = []
    
    def buildString(self, function=None, args=None):
        if function is None:
            function = self.functionalType
        if args is None:
            args = self.args
        
        if isinstance(function, FunctionalType):
            self.command = function.name
            self.functionalType = function
        else:
            self.command = function
            self.functionalType = FunctionalType.Other
        
        self.args = args
        self.command += ' ' + ', '.join(args)
        super().buildString()
    

    def is_load_const(self):
        if self.functionalType != FunctionalType.mov:
            return False
        if len(self.args) != 2:
            return False
        if self.args[0][0] != '%':
            return False
        if self.args[1][0] == '%':
            return False
        return True


class PTXEntryPoint (PTXLine):
    def __init__(self, fullLine = '') -> None:
        super().__init__(fullLine)


class PTXShader:
    def __init__(self, filePath) -> None:
        f = open(filePath, "r")
        lineNO = 1
        self.lines = []
        self.vectorVariables = list()
        for line in f:
            print('parsing line %s: %s' % (lineNO, line))
            ptxLine = PTXLine.createNewLine(line)
            if ptxLine.instructionClass == InstructionClass.VariableDeclaration and ptxLine.declarationType == DeclarationType.Register:
                if ptxLine.isVector():
                    self.vectorVariables.append(ptxLine.variableName)
            # print("#1")
            # print(line)
            # print(ptxLine.instructionClass)
            if ptxLine.instructionClass == InstructionClass.Functional:
                print(ptxLine.functionalType)
            # if ptxLine.instructionClass == InstructionClass.Functional:
            #     print(ptxLine.functionalType)
            self.lines.append(ptxLine)
            lineNO += 1
        # exit(-1)
        f.close()
    
    def findDeclaration(self, name):
        for index in range(len(self.lines)):
            line = self.lines[index]
            if line.instructionClass == InstructionClass.VariableDeclaration:
                if line.variableName == name:
                    return line, index
        return None, None
    
    def writeToFile(self, filePath):
        f = open(filePath, 'w')
        for line in self.lines:
            f.write(line.fullLine)
        f.close()
    
    def addToEndOfBlock(self, ptxLines, blockName):
        index = 0
        for index in range(len(self.lines)):
            line = self.lines[index]
            if blockName in line.fullLine and 'end_block' in line.fullLine:
                break
        
        while index > 0 and self.lines[index - 1].instructionClass == InstructionClass.Empty:
            index -= 1
        
        self.lines[index:index] = ptxLines
    

    def addToStartOfBlock(self, ptxLines, blockName):
        index = 0
        for index in range(len(self.lines)):
            line = self.lines[index]
            if blockName in line.fullLine and 'start_block' in line.fullLine:
                break
        
        while index < len(self.lines) - 1 and self.lines[index + 1].instructionClass == InstructionClass.Empty:
            index += 1
        
        self.lines[index:index] = ptxLines
    

    def addToStart(self, ptxLines):
        index = 0
        for index in range(len(self.lines)):
            line = self.lines[index]
            if 'start_block' in line.fullLine:
                break
        
        while index < len(self.lines) - 1 and self.lines[index + 1].instructionClass == InstructionClass.Empty:
            index += 1
        
        self.lines[index:index] = ptxLines