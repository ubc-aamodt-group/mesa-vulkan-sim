from enum import unique
from ptx_parser import *
import sys

def vector_suffix_letter(x):
    if x == 0:
        return 'x'
    elif x == 1:
        return 'y'
    elif x == 2:
        return 'z'
    elif x == 3:
        return 'w'

def vector_suffix_number(x):
    if x == 'x':
        return 0
    elif x == 'y':
        return 1
    elif x == 'z':
        return 2
    elif x == 'w':
        return 3



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
    # for i in range(declaration.vectorSize()):
    #     newMov = PTXFunctionalLine()
    #     newMov.leadingWhiteSpace = declaration.leadingWhiteSpace

    #     variableType = declaration.variableType
    #     if variableType == '.b32':
    #         variableType = '.f32'
    #     elif variableType == '.b64':
    #         variableType = '.f64'
    #     zero = '0'
    #     if variableType[1] == 'f':
    #         zero = '0F00000000'
    #     newMov.buildString('add%s' % variableType, (newRegNames[i], vectorVariableName + '.' + vector_suffix_letter(i), zero))
    #     unwrapMovs.append(newMov)
    newMov = PTXFunctionalLine()
    newMov.leadingWhiteSpace = declaration.leadingWhiteSpace
    newMov.buildString('unwrap_32_4', tuple(newRegNames + [vectorVariableName, ]))
    unwrapMovs.append(newMov)
    

    wrapMovs = list()
    # for i in range(declaration.vectorSize()):
    #     wrapMov = PTXFunctionalLine()
    #     wrapMov.leadingWhiteSpace = declaration.leadingWhiteSpace
    #     variableType = declaration.variableType
    #     if variableType == '.b32':
    #         variableType = '.f32'
    #     elif variableType == '.b64':
    #         variableType = '.f64'
    #     zero = '0'
    #     if variableType[1] == 'f':
    #         zero = '0F00000000'
    #     wrapMov.buildString('add%s' % (variableType), (vectorVariableName + '.' + vector_suffix_letter(i), newRegNames[i], zero))
    #     wrapMovs.append(wrapMov)
    wrapMov = PTXFunctionalLine()
    wrapMov.leadingWhiteSpace = declaration.leadingWhiteSpace
    wrapMov.buildString('wrap_32_4', tuple([vectorVariableName, ] + newRegNames))
    wrapMovs.append(wrapMov)

    return newRegNames, newDeclarations, unwrapMovs, wrapMovs


def translate_vector_operands(ptx_shader, unique_ID):
    index = -1
    while index + 1 < len(ptx_shader.lines):
        index += 1
        line = ptx_shader.lines[index]

        if line.instructionClass == InstructionClass.Functional:
            print("#######################")
            print(line.fullLine)

            for argIndex in range(len(line.args)):
                arg = line.args[argIndex]
                if '.' not in arg:
                    continue
                assert arg[-2] == '.'
                vectorRegName = arg[:-2]
                newRegName = vectorRegName + '_' + str(vector_suffix_number(arg[-1]))

                args = line.args
                args[argIndex] = newRegName
                line.buildString(line.fullFunction, args)
        
        elif line.instructionClass == InstructionClass.VariableDeclaration:
            if line.declarationType != DeclarationType.Register:
                continue
            if not line.isVector():
                continue

            print("#######################")
            print(line.fullLine)

            newLines = list()
            for i in range(line.vectorSize()):
                newLine = PTXDecleration()
                newLine.leadingWhiteSpace = line.leadingWhiteSpace
                newLine.buildString(line.declarationType, None, line.variableType, line.variableName + '_' + str(i))
                newLines.append(newLine)
            
            ptx_shader.lines.remove(line)
            ptx_shader.lines[index:index] = newLines



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
            dst, need_deref, src, arrayIndex, arrayStride, baseType, type = line.args

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
                newLines = list()

                # load into each register
                for i in range(declaration.vectorSize()):
                    newFunctional = PTXFunctionalLine()
                    newFunctional.leadingWhiteSpace = declaration.leadingWhiteSpace
                    newFunctional.buildString('ld.global%s' % declaration.variableType, (declaration.variableName + '_' + str(i), '[' + ptr + ' + ' + str(int(i * declaration.bitCount() / 8)) + ']'))
                    newLines.append(newFunctional)
                
                # insert the new lines into shader
                newLines.append(PTXLine('//' + line.comment + '\n'))
                ptx_shader.lines.remove(line)
                ptx_shader.lines[index: index] = newLines
        
        
        elif line.functionalType == FunctionalType.store_deref:
            # print("################")
            # print(line.fullLine)
            dst = line.args[1]
            ptr = line.args[0]

            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            if not declaration.isVector():
                line.buildString('st.global%s' % (declaration.variableType), ('[%s]' % ptr, dst))
            else:
                newLines = list()

                # load into each register
                for i in range(declaration.vectorSize()):
                    newFunctional = PTXFunctionalLine()
                    newFunctional.leadingWhiteSpace = declaration.leadingWhiteSpace
                    newFunctional.buildString('st.global%s' % declaration.variableType, ('[' + ptr  + ' + ' + str(int(i * declaration.bitCount() / 8)) + ']', declaration.variableName + '_' + str(i)))
                    newLines.append(newFunctional)
                
                # insert the new lines into shader
                ptx_shader.lines.remove(line)
                newLines.append(PTXLine('//' + line.comment + '\n'))
                ptx_shader.lines[index: index] = newLines
        

        elif line.functionalType == FunctionalType.deref_var:
            dst, src, type = line.args
            
            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            assert not declaration.isVector()
            assert declaration.declarationType == DeclarationType.Register
            declaration.buildString(DeclarationType.Register, None, '.b64', declaration.variableName)

            line.buildString('mov.b64', (dst, '%' + src))


        # elif line.functionalType == FunctionalType.mov:
        #     print(line.fullLine)
        #     print(line.args)
        #     #assert len(line.args) == 2
        #     if '.' in line.args[0]: #TODO: args with brackets are parsed incorrectly
        #         if line.vector == None:
        #             variableType = line.variableType
        #             print(line.fullLine)
        #             print(variableType)
        #             #exit(-1)
        #             if variableType == '.b32':
        #                 variableType = '.f32'
        #             elif variableType == '.b64':
        #                 variableType = '.f64'
        #             zero = '0'
        #             if variableType[1] == 'f':
        #                 zero = '0F00000000'
        #             line.buildString('add%s' % variableType, (line.args[0], line.args[1], zero))

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
        line.args = line.args[:-1] # MRS_TODO: why there is a payload (in glsl code it is NULL but translated to ssa_88)? and why it gets an error to run?
        args = line.args

        # originRegNames, originDeclarations, originMovs, _ = unwrapp_vector(ptx_shader, origin, "trace_ray_" + str(index) + "_origin")
        # directionRegNames, directionDeclarations, directionMovs, _ = unwrapp_vector(ptx_shader, direction, "trace_ray_" + str(index) + "_direction")

        originRegNames = [origin + '_' + str(i) for i in range(3)]
        directionRegNames = [direction + '_' + str(i) for i in range(3)]

        hitDeclaration = PTXDecleration()
        hitDeclaration.leadingWhiteSpace = line.leadingWhiteSpace
        hitDeclaration.buildString(DeclarationType.Register, None, '.u32', 'hit_geometry')

        args[8:9] = directionRegNames[:3]
        args[6:7] = originRegNames[:3]
        args.append('hit_geometry')
        line.buildString(line.functionalType, args)

        call_closest_hit = PTXFunctionalLine()
        call_closest_hit.leadingWhiteSpace = line.leadingWhiteSpace
        call_closest_hit.condition = '@!hit_geometry'
        call_closest_hit.buildString(FunctionalType.call_closest_hit_shader, ())

        call_miss = PTXFunctionalLine()
        call_miss.leadingWhiteSpace = line.leadingWhiteSpace
        call_miss.condition = '@hit_geometry'
        call_miss.buildString(FunctionalType.call_miss_shader, ())

        ptx_shader.lines.insert(index, hitDeclaration)
        # ptx_shader.lines[index + 1:index + 1] = originDeclarations
        # ptx_shader.lines[index + 5:index + 5] = directionDeclarations
        # ptx_shader.lines[index + 9:index + 9] = originMovs[:3]
        # ptx_shader.lines[index + 12:index + 12] = directionMovs[:3]
        ptx_shader.lines.insert(index + 2, call_closest_hit)
        ptx_shader.lines.insert(index + 3, call_miss)

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

        # print(line.fullLine)
        # print(line.args)
        # exit(-1)

        name, bit_size, vector_number, variable_type, storage_qualifier_type, driver_location, binding = line.args
        name = '%' + name

        newReg = PTXDecleration()
        newReg.leadingWhiteSpace = '\t'
        newReg.buildString(DeclarationType.Register, None, '.b64', name)


        if int(storage_qualifier_type) == 16: ## uniform type
            newLine = PTXFunctionalLine()
            newLine.leadingWhiteSpace = '\t'
            newLine.comment = line.comment
            newLine.buildString('load_vulkan_descriptor', (name, driver_location, binding))
        else:
            newLine = PTXFunctionalLine()
            newLine.leadingWhiteSpace = '\t'
            newLine.comment = line.comment
            newLine.buildString('rt_alloc_mem', (name, str(int(int(bit_size) * int(vector_number) / 8)), str(storage_qualifier_type)))

        new_declerations.append(newReg)
        # new_declerations.append(newSizeSet)
        new_declerations.append(newLine)
        old_declerations.append(line)
    
    for decleration in old_declerations:
        ptx_shader.lines.remove(decleration)
    

    ptx_shader.addToStart(new_declerations)
    
    # for index in range(len(ptx_shader.lines)):
    #     line = ptx_shader.lines[index]
    #     if line.instructionClass != InstructionClass.EntryPoint:
    #         continue
    #     if 'main' not in line.fullLine:
    #         continue
        
    #     index += 1
    #     for decleration in new_declerations:
    #         ptx_shader.lines.insert(index + 1, decleration)
    #         index += 1
    #     break


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

            # newDeclarations = list()
            # for i in range(4):
            #     newDeclaration = PTXDecleration()
            #     newDeclaration.leadingWhiteSpace = line.leadingWhiteSpace
            #     newDeclaration.buildString(DeclarationType.Register, None, declaration.variableType, newRegNames[i])
            #     newDeclarations.append(newDeclaration)

            # comment = line.comment
            # line.comment = ""
            line.buildString(line.functionalType, (newRegNames[:3]))

            # loadZero = PTXFunctionalLine()
            # loadZero.leadingWhiteSpace = line.leadingWhiteSpace
            # loadZero.buildString('mov%s' % (declaration.variableType), (newRegNames[3], "0"))
            
            # _, _, _, wrapMovs = unwrapp_vector(ptx_shader, declaration.variableName, declaration.variableName)
            # movLine = PTXFunctionalLine()
            # movLine.leadingWhiteSpace = line.leadingWhiteSpace
            # movLine.comment = comment
            # movLine.buildString('mov%s%s' % (declaration.vector, declaration.variableType), (declaration.variableName, '{' + ", ".join(newRegNames) + '}'))

            # ptx_shader.lines[index:index] =  newDeclarations
            # ptx_shader.lines.insert(index + 5, loadZero)
            # ptx_shader.lines[index + 6: index + 6] = wrapMovs[:3]
            # skip_lines = index + 7

def translate_image_deref_store(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]

        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType != FunctionalType.image_deref_store:
            continue

        image, arg2, arg3, hitValue, arg5, arg6, arg7 = line.args
        args = line.args
        args[3:4] = [(hitValue + '_' + str(i)) for i in range(4)]
        args[1:2] = [(arg2 + '_' + str(i)) for i in range(4)]
        line.buildString(line.functionalType, args)


def translate_exit(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]

        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType != FunctionalType.exit:
            continue

        line.buildString(FunctionalType.ret, ())


def translate_phi(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType != FunctionalType.phi:
            continue

        dst, blockName0, src0, blockName1, src1 = line.args
        
        dstDecleration, dstIndex = ptx_shader.findDeclaration(dst)
        src0Decleration, _ = ptx_shader.findDeclaration(src0)
        src1Decleration, _ = ptx_shader.findDeclaration(src1)

        if src0Decleration.variableType == src1Decleration.variableType:
            variableType = src0Decleration.variableType
        else: # this happens because of load_const types are unknown
            if src0Decleration.variableType[0:2] == '.f':
                variableType = src1Decleration.variableType
            elif src1Decleration.variableType[0:2] == '.f':
                variableType = src0Decleration.variableType
            else:
                assert 0

        dstDecleration.buildString(dstDecleration.declarationType, dstDecleration.vector, variableType, dstDecleration.variableName)

        src0Mov = PTXFunctionalLine()
        src0Mov.leadingWhiteSpace = src0Decleration.leadingWhiteSpace
        src0Mov.comment = line.comment
        src0Mov.buildString('mov%s' % variableType, (dst, src0))

        src1Mov = PTXFunctionalLine()
        src1Mov.leadingWhiteSpace = src1Decleration.leadingWhiteSpace
        src1Mov.comment = line.comment
        src1Mov.buildString('mov%s' % variableType, (dst, src1))


        ptx_shader.lines.remove(dstDecleration)
        ptx_shader.lines.remove(line)

        ptx_shader.addToStart((dstDecleration, ))
        ptx_shader.addToEndOfBlock((src0Mov, ), blockName0)
        ptx_shader.addToEndOfBlock((src1Mov, ), blockName1)



def main():
    unique_ID = 0
    assert len(sys.argv) == 2
    # shaderPath = sys.argv[1]
    # shader = PTXShader(shaderPath)
    # translate_descriptor_set_instructions(shader)
    # translate_deref_instructions(shader)
    # translate_trace_ray(shader)
    # translate_decl_var(shader)
    # translate_ray_launch_instructions(shader)
    # translate_image_deref_store(shader)
    # translate_exit(shader)
    # translate_phi(shader)

    # translate_vector_operands(shader, unique_ID)
    # shader.writeToFile(shaderPath)


main()