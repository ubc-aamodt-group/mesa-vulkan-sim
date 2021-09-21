from ptx_parser import *
import sys

def vector_suffix(x):
    if x == 0:
        return 'x'
    elif x == 1:
        return 'y'
    elif x == 2:
        return 'z'
    elif x == 3:
        return 'w'


def unwrapp_vector(ptx_shader, vectorVariableName, unwrappedName):
    declaration, _ = ptx_shader.findDeclaration(vectorVariableName)
    assert declaration.isVector()

    newRegNames = [(unwrappedName + '_' + str(i)) for i in range(declaration.vectorSize())]

    newDeclarations = list()
    for i in range(declaration.vectorSize()):
        newDeclaration = PTXDecleration()
        newDeclaration.leadingWhiteSpace = declaration.leadingWhiteSpace
        newDeclaration.buildString(DeclarationType.Register, None, declaration.variableType, newRegNames[i])
        newDeclarations.append(newDeclaration)

    unwrapMovs = list()
    for i in range(declaration.vectorSize()):
        newMov = PTXFunctionalLine()
        newMov.leadingWhiteSpace = declaration.leadingWhiteSpace

        variableType = declaration.variableType
        if variableType == '.b32':
            variableType = '.f32'
        elif variableType == '.b64':
            variableType = '.f64'
        zero = '0'
        if variableType[1] == 'f':
            zero = '0F00000000'
        newMov.buildString('add%s' % variableType, (newRegNames[i], vectorVariableName + '.' + vector_suffix(i), zero))
        unwrapMovs.append(newMov)
    

    wrapMovs = list()
    for i in range(declaration.vectorSize()):
        wrapMov = PTXFunctionalLine()
        wrapMov.leadingWhiteSpace = declaration.leadingWhiteSpace
        variableType = declaration.variableType
        if variableType == '.b32':
            variableType = '.f32'
        elif variableType == '.b64':
            variableType = '.f64'
        zero = '0'
        if variableType[1] == 'f':
            zero = '0F00000000'
        wrapMov.buildString('add%s' % (variableType), (vectorVariableName + '.' + vector_suffix(i), newRegNames[i], zero))
        wrapMovs.append(wrapMov)

    return newRegNames, newDeclarations, unwrapMovs, wrapMovs


def translate_descriptor_set_instructions(ptx_shader):
    for line in ptx_shader.lines:
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.vulkan_resource_index:
            dst, whatIsThis, descSet, binding, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)

            line.buildString(FunctionalType.load_vulkan_descriptor, (dst, descSet, binding, type))
        
        elif line.functionalType == FunctionalType.load_vulkan_descriptor:
            dst, src, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)

            line.buildString('mov.b64', (dst, src))

def translate_deref_instructions(ptx_shader):
    index = -1
    while index + 1 < len(ptx_shader.lines):
        index += 1
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.deref_cast:
            dst, structName, src, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)

            line.buildString('mov.b64', (dst, src))
        
        elif line.functionalType == FunctionalType.deref_struct:
            dst, need_deref, src, ptr, fieldName, offset, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)

            line.buildString('add.u64', (dst, src, offset))
        
        elif line.functionalType == FunctionalType.deref_array:
            dst, need_deref, src, arrayIndex, arrayStride, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)

            line.buildString('add.u64', (dst, src, str(int(arrayIndex) * int(arrayStride))))
        
        elif line.functionalType == FunctionalType.load_deref:
            dst = line.args[0]
            ptr = line.args[1]

            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            if not declaration.isVector():
                line.buildString('ld.global%s' % (declaration.variableType), (dst, '[%s]' % ptr))
            else:
                line.buildString('ld.global%s%s' % (declaration.vector if declaration.vector is not None else '', declaration.variableType), (dst, '[%s]' % ptr))
                newLines = list()
                
                # declare new register to load individually
                declarationName = ''
                for i in range(declaration.vectorSize()):
                    declarationName += declaration.variableName + '_' + str(i)
                    if i != declaration.vectorSize() - 1:
                        declarationName += ', '
                newDecleration = PTXDecleration()
                newDecleration.leadingWhiteSpace = declaration.leadingWhiteSpace
                newDecleration.buildString(DeclarationType.Register, None, declaration.variableType, declarationName)
                newLines.append(newDecleration)

                # load into each register
                for i in range(declaration.vectorSize()):
                    newFunctional = PTXFunctionalLine()
                    newFunctional.leadingWhiteSpace = declaration.leadingWhiteSpace
                    newFunctional.buildString('ld.global%s' % declaration.variableType, (declaration.variableName + '_' + str(i), '[' + ptr + ' + ' + str(int(i * declaration.bitCount() / 8)) + ']'))
                    newLines.append(newFunctional)
                
                _, _, _, wrapMovs = unwrapp_vector(ptx_shader, declaration.variableName, declaration.variableName)
                
                # insert the new lines into shader
                ptx_shader.lines[index] = PTXLine('//' + line.comment + '\n')
                ptx_shader.lines[declerationLine + 1: declerationLine + 1] = wrapMovs
                ptx_shader.lines[declerationLine: declerationLine] = newLines #TODO: why not store in current location?

                # mov indicidual registers to final vector register
                # line.buildString('mov%s%s' % (declaration.vector, declaration.variableType), (declaration.variableName, '{' + ', '.join([declaration.variableName + '_' + str(i) for i in range(declaration.vectorSize())]) + '}'))
        
        
        elif line.functionalType == FunctionalType.store_deref:
            dst = line.args[1]
            ptr = line.args[0]

            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            if not declaration.isVector():
                line.buildString('st.global%s' % (declaration.variableType), ('[%s]' % ptr, dst))
            else:
                line.buildString('st.global%s%s' % (declaration.vector if declaration.vector is not None else '', declaration.variableType), ('[%s]' % ptr, dst))
                newLines = list()
                
                # declare new register to load individually
                declarationName = ''
                for i in range(declaration.vectorSize()):
                    declarationName += declaration.variableName + '_' + str(i)
                    if i != declaration.vectorSize() - 1:
                        declarationName += ', '
                newDecleration = PTXDecleration()
                newDecleration.leadingWhiteSpace = declaration.leadingWhiteSpace
                newDecleration.buildString(DeclarationType.Register, None, declaration.variableType, declarationName)
                newLines.append(newDecleration)

                _, _, unwrapMovs, _ = unwrapp_vector(ptx_shader, declaration.variableName, declaration.variableName)
                newLines = newLines + unwrapMovs

                # load into each register
                for i in range(declaration.vectorSize()):
                    newFunctional = PTXFunctionalLine()
                    newFunctional.leadingWhiteSpace = declaration.leadingWhiteSpace
                    newFunctional.buildString('st.global%s' % declaration.variableType, ('[' + ptr  + ' + ' + str(int(i * declaration.bitCount() / 8)) + ']', declaration.variableName + '_' + str(i)))
                    newLines.append(newFunctional)
                
                # insert the new lines into shader
                ptx_shader.lines[index] = PTXLine('//' + line.comment + '\n')
                ptx_shader.lines[index: index] = newLines

                # mov indicidual registers to final vector register
                # line.buildString('mov%s%s' % (declaration.vector, declaration.variableType), (declaration.variableName, '{' + ', '.join([declaration.variableName + '_' + str(i) for i in range(declaration.vectorSize())]) + '}'))
        
        # elif line.functionalType == FunctionalType.deref_var:
        #     print('@#^$*@^#&$*@#&^$* got here')
        #     print(line.fullLine)
        #     line.fullLine = '//' + line.fullLine
        #     print(line.fullLine)

        elif line.functionalType == FunctionalType.mov:
            print(line.fullLine)
            print(line.args)
            #assert len(line.args) == 2
            if '.' in line.args[0]: #TODO: args with brackets are parsed incorrectly
                if line.vector == None:
                    variableType = line.variableType
                    print(line.fullLine)
                    print(variableType)
                    #exit(-1)
                    if variableType == '.b32':
                        variableType = '.f32'
                    elif variableType == '.b64':
                        variableType = '.f64'
                    zero = '0'
                    if variableType[1] == 'f':
                        zero = '0F00000000'
                    line.buildString('add%s' % variableType, (line.args[0], line.args[1], zero))

def translate_trace_ray(ptx_shader):
    skip_lines = -1
    for index in range(len(ptx_shader.lines)):
        if index <= skip_lines:
            continue
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType != FunctionalType.trace_ray:
            continue

        assert len(line.args) == 11

        topLevelAS, rayFlags, cullMask, sbtRecordOffset, sbtRecordStride, missIndex, origin, Tmin, direction, Tmax, payload = line.args
        args = line.args

        originRegNames, originDeclarations, originMovs, _ = unwrapp_vector(ptx_shader, origin, "trace_ray_" + str(index) + "_origin")
        directionRegNames, directionDeclarations, directionMovs, _ = unwrapp_vector(ptx_shader, direction, "trace_ray_" + str(index) + "_direction")

        hitDeclaration = PTXDecleration()
        hitDeclaration.leadingWhiteSpace = line.leadingWhiteSpace
        hitDeclaration.buildString(DeclarationType.Register, None, '.pred', 'hit')

        args[8:9] = directionRegNames[:3]
        args[6:7] = originRegNames[:3]
        line.buildString(line.functionalType, args)

        call_closest_hit = PTXFunctionalLine()
        call_closest_hit.leadingWhiteSpace = line.leadingWhiteSpace
        call_closest_hit.buildString(FunctionalType.call_closest_hit_shader, ())
        call_closest_hit.condition = '@hit'

        call_miss = PTXFunctionalLine()
        call_miss.leadingWhiteSpace = line.leadingWhiteSpace
        call_miss.buildString(FunctionalType.call_miss_shader, ())
        call_miss.condition = '@!hit'

        ptx_shader.lines.insert(index, hitDeclaration)
        ptx_shader.lines[index + 1:index + 1] = originDeclarations[:3]
        ptx_shader.lines[index + 4:index + 4] = directionDeclarations[:3]
        ptx_shader.lines[index + 7:index + 7] = originMovs[:3]
        ptx_shader.lines[index + 10:index + 10] = directionMovs[:3]
        ptx_shader.lines.insert(index + 14, call_closest_hit)
        ptx_shader.lines.insert(index + 15, call_miss)

        print(ptx_shader.lines[index + 2].fullLine)

        skip_lines = index + 16


def translate_decl_var(ptx_shader):
    new_declerations = []
    old_declerations = []

    # newReg = PTXDecleration()
    # newReg.leadingWhiteSpace = '\t'
    # newReg.buildString(DeclarationType.Register, None, '.u32', '%allocasize')
    # new_declerations.append(newReg)


    for line in ptx_shader.lines:
        if line.instructionClass != InstructionClass.Functional:
            continue
        if line.functionalType != FunctionalType.decl_var:
            continue

        name, bit_size, vector_number, variable_type, storage_qualifier_type = line.args
        name = '%' + name

        newReg = PTXDecleration()
        newReg.leadingWhiteSpace = '\t'
        newReg.buildString(DeclarationType.Register, None, '.u64', name)

        # newSizeSet = PTXFunctionalLine()
        # newSizeSet.leadingWhiteSpace = '\t'
        # newSizeSet.buildString('mov.u32', ('%allocasize', str(int(int(bit_size) / 8))))

        newLine = PTXFunctionalLine()
        newLine.leadingWhiteSpace = '\t'
        newLine.comment = line.comment
        newLine.buildString('rt_alloc_mem', (name, str(int(int(bit_size) / 8))))

        new_declerations.append(newReg)
        # new_declerations.append(newSizeSet)
        new_declerations.append(newLine)
        old_declerations.append(line)
    
    for decleration in old_declerations:
        ptx_shader.lines.remove(decleration)
    
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.EntryPoint:
            continue
        if 'main' not in line.fullLine:
            continue
        
        index += 1
        for decleration in new_declerations:
            ptx_shader.lines.insert(index + 1, decleration)
            index += 1
        break


def  translate_ray_launch_instructions(ptx_shader):
    skip_lines = -1
    for index in range(len(ptx_shader.lines)):
        if index <= skip_lines:
            continue
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue
        if line.functionalType == FunctionalType.load_ray_launch_id or line.functionalType == FunctionalType.load_ray_launch_size:
            dst = line.args[0]

            declaration, _ = ptx_shader.findDeclaration(dst)
            assert declaration.isVector()

            newRegNames = [(dst + '_' + str(i)) for i in range(4)]

            newDeclarations = list()
            for i in range(4):
                newDeclaration = PTXDecleration()
                newDeclaration.leadingWhiteSpace = line.leadingWhiteSpace
                newDeclaration.buildString(DeclarationType.Register, None, declaration.variableType, newRegNames[i])
                newDeclarations.append(newDeclaration)

            comment = line.comment
            line.comment = ""
            line.buildString(line.functionalType, (newRegNames[:3]))

            loadZero = PTXFunctionalLine()
            loadZero.leadingWhiteSpace = line.leadingWhiteSpace
            loadZero.buildString('mov%s' % (declaration.variableType), (newRegNames[3], "0"))
            
            _, _, _, wrapMovs = unwrapp_vector(ptx_shader, declaration.variableName, declaration.variableName)
            # movLine = PTXFunctionalLine()
            # movLine.leadingWhiteSpace = line.leadingWhiteSpace
            # movLine.comment = comment
            # movLine.buildString('mov%s%s' % (declaration.vector, declaration.variableType), (declaration.variableName, '{' + ", ".join(newRegNames) + '}'))

            ptx_shader.lines[index:index] =  newDeclarations
            ptx_shader.lines.insert(index + 5, loadZero)
            ptx_shader.lines[index + 6: index + 6] = wrapMovs[:3]
            skip_lines = index + 7



def main():
    assert len(sys.argv) == 2
    shaderPath = sys.argv[1]
    shader = PTXShader(shaderPath)
    translate_descriptor_set_instructions(shader)
    translate_deref_instructions(shader)
    translate_trace_ray(shader)
    translate_decl_var(shader)
    translate_ray_launch_instructions(shader)
    shader.writeToFile(shaderPath)
main()