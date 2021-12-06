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
            # print("#######################")
            # print(line.fullLine)

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

            # print("#######################")
            # print(line.fullLine)

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
            dst, baseType, src, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)
            declaration.pointerVariableType = '.' + baseType

            line.buildString('mov.b64', (dst, src))
        
        elif line.functionalType == FunctionalType.deref_struct:
            dst, need_deref, src, ptr, fieldName, offset, baseType, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)
            declaration.pointerVariableType = '.' + baseType

            line.buildString('add.u64', (dst, src, offset))
        
        elif line.functionalType == FunctionalType.deref_array:
            dst, need_deref, src, arrayIndex, arrayStride, baseType, type = line.args

            declaration, _ = ptx_shader.findDeclaration(dst)
            declaration.buildString(declaration.declarationType, None, '.b64', declaration.variableName)
            declaration.pointerVariableType = '.' + baseType

            if arrayIndex[0] != '%': # const array index
                line.buildString('add.u64', (dst, src, str(int(arrayIndex) * int(arrayStride))))
            else: # reg array index

                # exit(-1)
                newLines = list()

                indexRegName_32 = dst + '_array_index_32'
                indexRegName_64 = dst + '_array_index_64'

                newDeclaration_32 = PTXDecleration()
                newDeclaration_32.leadingWhiteSpace = declaration.leadingWhiteSpace
                newDeclaration_32.buildString(DeclarationType.Register, None, '.u32', indexRegName_32)

                newDeclaration_64 = PTXDecleration()
                newDeclaration_64.leadingWhiteSpace = declaration.leadingWhiteSpace
                newDeclaration_64.buildString(DeclarationType.Register, None, '.u64', indexRegName_64)

                indexDeclaration, _ = ptx_shader.findDeclaration(arrayIndex)

                newSet = PTXFunctionalLine()
                newSet.leadingWhiteSpace = declaration.leadingWhiteSpace
                if indexDeclaration.variableType == '.u32':
                    newSet.buildString('mov.u32', (indexRegName_32, arrayIndex))
                else:
                    variableType = indexDeclaration.variableType
                    if variableType[1] == 'b':
                        variableType = '.u' + variableType[2:]
                    newSet.buildString('cvt.u32%s' % variableType, (indexRegName_32, arrayIndex))


                newMult = PTXFunctionalLine()
                newMult.leadingWhiteSpace = declaration.leadingWhiteSpace
                newMult.buildString('mul.wide.u32', (indexRegName_64, indexRegName_32, arrayStride))

                newAdd = PTXFunctionalLine()
                newAdd.leadingWhiteSpace = declaration.leadingWhiteSpace
                newAdd.comment = line.comment
                newAdd.buildString('add.u64', (dst, src, indexRegName_64))

                ptx_shader.lines.remove(line)
                ptx_shader.lines[index:index] = (newDeclaration_32, newDeclaration_64, newSet, newMult, newAdd)

        
        elif line.functionalType == FunctionalType.load_deref:
            vectorCount, dst, ptr, access = line.args

            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            srcDeclaration, _ = ptx_shader.findDeclaration(ptr)

            # print(line.fullLine)

            # assert srcDeclaration.pointerVariableType[:2] == '.b' or srcDeclaration.pointerVariableType[:2] == '.f'

            assert srcDeclaration.pointerVariableType[2:] == declaration.variableType[2:]
            declaration.buildString(declaration.declarationType, declaration.vector, srcDeclaration.pointerVariableType, declaration.variableName)

            # if srcDeclaration.pointerVariableType == None:
            #     exit(-1)

            if not declaration.isVector():
                line.buildString('ld.global%s' % (declaration.variableType), (dst, '[%s]' % ptr))
            else:
                newLines = list()

                # load into each register
                for i in range(declaration.vectorSize()):
                    if int(vectorCount) > 0 and i >= int(vectorCount):
                        break
                    newFunctional = PTXFunctionalLine()
                    newFunctional.leadingWhiteSpace = declaration.leadingWhiteSpace
                    # print('#432 ' + declaration.fullLine)
                    newFunctional.buildString('ld.global%s' % declaration.variableType, (declaration.variableName + '_' + str(i), '[' + ptr + ' + ' + str(int(i * declaration.bitCount() / 8)) + ']'))
                    newLines.append(newFunctional)
                
                # insert the new lines into shader
                newLines.append(PTXLine('//' + line.comment + '\n'))
                ptx_shader.lines.remove(line)
                ptx_shader.lines[index: index] = newLines
        
        
        elif line.functionalType == FunctionalType.store_deref:
            # print("################")
            # print(line.fullLine)
            ptr, dst, wrmask, access = line.args
            # dst = line.args[1]
            # ptr = line.args[0]

            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            if not declaration.isVector():
                line.buildString('st.global%s' % (declaration.variableType), ('[%s]' % ptr, dst))
            else:
                newLines = list()

                # load into each register
                for i in range(declaration.vectorSize()):
                    if int(wrmask) & (1 << i) == 0:
                        continue
                    newFunctional = PTXFunctionalLine()
                    newFunctional.leadingWhiteSpace = declaration.leadingWhiteSpace
                    newFunctional.buildString('st.global%s' % declaration.variableType, ('[' + ptr  + ' + ' + str(int(i * declaration.bitCount() / 8)) + ']', declaration.variableName + '_' + str(i)))
                    newLines.append(newFunctional)
                
                # insert the new lines into shader
                ptx_shader.lines.remove(line)
                newLines.append(PTXLine('//' + line.comment + '\n'))
                ptx_shader.lines[index: index] = newLines
        

        elif line.functionalType == FunctionalType.deref_var:
            dst, src, baseType, type = line.args
            
            declaration, declerationLine = ptx_shader.findDeclaration(dst)
            assert not declaration.isVector()
            assert declaration.declarationType == DeclarationType.Register
            declaration.buildString(DeclarationType.Register, None, '.b64', declaration.variableName)
            declaration.pointerVariableType = '.' + baseType

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

        runClosestHitDeclaration = PTXDecleration()
        runClosestHitDeclaration.leadingWhiteSpace = line.leadingWhiteSpace
        runClosestHitDeclaration.buildString(DeclarationType.Register, None, '.u32', '%run_closest_hit')

        runMissDeclaration = PTXDecleration()
        runMissDeclaration.leadingWhiteSpace = line.leadingWhiteSpace
        runMissDeclaration.buildString(DeclarationType.Register, None, '.u32', '%run_miss')

        args[8:9] = directionRegNames[:3]
        args[6:7] = originRegNames[:3]
        args.append('%run_closest_hit')
        args.append('%run_miss')
        line.buildString(line.functionalType, args)


        skip_closest_hit_declaration = PTXDecleration()
        skip_closest_hit_declaration.leadingWhiteSpace = line.leadingWhiteSpace
        skip_closest_hit_declaration.buildString(DeclarationType.Register, None, '.pred', '%skip_closest_hit')

        call_closest_hit_setp = PTXFunctionalLine()
        call_closest_hit_setp.leadingWhiteSpace = line.leadingWhiteSpace
        call_closest_hit_setp.buildString('setp.eq.u32', ('%skip_closest_hit', '%run_closest_hit', '%const0_u32'))

        call_closest_hit_bra = PTXFunctionalLine()
        call_closest_hit_bra.leadingWhiteSpace = line.leadingWhiteSpace
        call_closest_hit_bra.condition = '@%skip_closest_hit'
        call_closest_hit_bra.buildString(FunctionalType.bra, ('skip_closest_hit_label', ))

        call_closest_hit = PTXFunctionalLine()
        call_closest_hit.leadingWhiteSpace = line.leadingWhiteSpace
        call_closest_hit.buildString(FunctionalType.call_closest_hit_shader, ())

        skip_closest_hit_label = PTXLine('')
        skip_closest_hit_label.fullLine = line.leadingWhiteSpace + 'skip_closest_hit_label:\n'


        skip_miss_declaration = PTXDecleration()
        skip_miss_declaration.leadingWhiteSpace = line.leadingWhiteSpace
        skip_miss_declaration.buildString(DeclarationType.Register, None, '.pred', '%skip_miss')

        call_miss_setp = PTXFunctionalLine()
        call_miss_setp.leadingWhiteSpace = line.leadingWhiteSpace
        call_miss_setp.buildString('setp.eq.u32', ('%skip_miss', '%run_miss', '%const0_u32'))
        
        call_miss_bra = PTXFunctionalLine()
        call_miss_bra.leadingWhiteSpace = line.leadingWhiteSpace
        call_miss_bra.condition = '@%skip_miss'
        call_miss_bra.buildString(FunctionalType.bra, ('skip_miss_label', ))

        call_miss = PTXFunctionalLine()
        call_miss.leadingWhiteSpace = line.leadingWhiteSpace
        call_miss.buildString(FunctionalType.call_miss_shader, ())

        skip_miss_label = PTXLine('')
        skip_miss_label.fullLine = line.leadingWhiteSpace + 'skip_miss_label:\n'


        end_trace_ray = PTXFunctionalLine()
        end_trace_ray.leadingWhiteSpace = line.leadingWhiteSpace
        end_trace_ray.buildString(FunctionalType.end_trace_ray, ())


        ptx_shader.lines[index:index + 1] = (runClosestHitDeclaration, runMissDeclaration, line, PTXLine('\n'), \
            skip_closest_hit_declaration, call_closest_hit_setp, call_closest_hit_bra, call_closest_hit, skip_closest_hit_label, PTXLine('\n'),\
            skip_miss_declaration, call_miss_setp, call_miss_bra, call_miss, skip_miss_label, PTXLine('\n'),\
            end_trace_ray)
        
        skip_lines = index + 16


        # call_closest_hit = PTXFunctionalLine()
        # call_closest_hit.leadingWhiteSpace = line.leadingWhiteSpace
        # call_closest_hit.condition = '@!%run_closest_hit'
        # call_closest_hit.buildString(FunctionalType.call_closest_hit_shader, ())

        # call_miss = PTXFunctionalLine()
        # call_miss.leadingWhiteSpace = line.leadingWhiteSpace
        # call_miss.condition = '@!%run_miss'
        # call_miss.buildString(FunctionalType.call_miss_shader, ())

        # ptx_shader.lines.insert(index, runClosestHitDeclaration)
        # ptx_shader.lines.insert(index + 1, runMissDeclaration)
        # ptx_shader.lines.insert(index + 3, call_closest_hit)
        # ptx_shader.lines.insert(index + 4, call_miss)

        # skip_lines = index + 16


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

        name, size, vector_number, variable_type, storage_qualifier_type, driver_location, binding = line.args
        name = '%' + name

        newReg = PTXDecleration()
        newReg.leadingWhiteSpace = '\t'
        newReg.buildString(DeclarationType.Register, None, '.b64', name)

        if int(vector_number) != 0:
            allocation_size = int(size) * int(vector_number)
        else:
            allocation_size = int(size)


        if int(storage_qualifier_type) == 16: ## uniform type
            newLine = PTXFunctionalLine()
            newLine.leadingWhiteSpace = '\t'
            newLine.comment = line.comment
            newLine.buildString('load_vulkan_descriptor', (name, driver_location, binding))
        else:
            newLine = PTXFunctionalLine()
            newLine.leadingWhiteSpace = '\t'
            newLine.comment = line.comment
            newLine.buildString('rt_alloc_mem', (name, str(allocation_size), str(storage_qualifier_type)))

        new_declerations.append(newReg)
        # new_declerations.append(newSizeSet)
        new_declerations.append(newLine)
        old_declerations.append(line)
    
    for decleration in old_declerations:
        ptx_shader.lines.remove(decleration)
    
    new_declerations.append(PTXLine('\n'))
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


def translate_load_GL_instructions(ptx_shader):
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
        
        
        elif line.functionalType == FunctionalType.load_ray_world_to_object or line.functionalType == FunctionalType.load_ray_object_to_world:
            dst, loadIndex = line.args

            newRegNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)

            line.buildString(line.functionalType, newRegNames[:3] + [loadIndex, ])
        

        elif line.functionalType == FunctionalType.load_ray_world_direction:
            dst = line.args[0]

            newRegNames, _, _, _ = unwrapp_vector(ptx_shader, dst, dst)
            line.buildString(line.functionalType, newRegNames[:3])







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

        ptx_shader.addToStart((dstDecleration, PTXLine('\n')))
        ptx_shader.addToEndOfBlock((src0Mov, ), blockName0)
        ptx_shader.addToEndOfBlock((src1Mov, ), blockName1)


def translate_load_const(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType != FunctionalType.load_const:
            continue

        dst, const = line.args
        declaration, _ = ptx_shader.findDeclaration(dst)
        declaration.isLoadConst = True

        line.buildString("mov%s" % (declaration.variableType), (dst, const))


        newVariableType = '.b' + declaration.variableType[2:]

        newDeclaration = PTXDecleration()
        newDeclaration.leadingWhiteSpace = declaration.leadingWhiteSpace
        newDeclaration.buildString(DeclarationType.Register, None, newVariableType, dst + '_bits')

        newMov = PTXFunctionalLine()
        newMov.leadingWhiteSpace = line.leadingWhiteSpace
        newMov.buildString("mov%s" % (declaration.variableType), (dst + '_bits', const))

        ptx_shader.lines[index + 1:index + 1] = (newDeclaration, newMov)




def translate_const_operands(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.command[:3] == 'mov':

            movType = line.command[3:]

            dst, src = line.args

            print(line.fullLine)
            
            if src[0] == '%':
                declaration, _ = ptx_shader.findDeclaration(src)                

                if movType[:2] != '.f' and declaration.isLoadConst:
                    line.buildString(line.command.split()[0], (dst, src + '_bits'))
        
        elif line.command[:4] == 'setp':
            dst, src1, src2 = line.args

            type = '.' + line.command.split()[0].split('.')[2]

            if type[:2] != '.f':
                if src1[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src1)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1 + '_bits', src2))
            
                if src2[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src2)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1, src2 + '_bits'))
        
        elif line.command[:3] == 'add':
            dst, src1, src2 = line.args

            type = line.command[3:]

            if type[:2] != '.f':
                if src1[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src1)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1 + '_bits', src2))
            
                if src2[0] == '%':
                    declaration, _ = ptx_shader.findDeclaration(src2)
                    if declaration.isLoadConst:
                        line.buildString(line.command.split()[0], (dst, src1, src2 + '_bits'))


        
def translate_f1_to_pred(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        # if line.instructionClass == InstructionClass.VariableDeclaration:

        #     if line.declarationType == DeclarationType.Register:

        #         if line.variableType == '.f1':
        #             line.buildString(DeclarationType.Register, line.vector, '.pred', line.variableName)
        

        if line.instructionClass == InstructionClass.Functional:

            if line.command.split()[0] == 'ld.global.b1':
                
                dst, ptr = line.args
                declaration, _ = ptx_shader.findDeclaration(dst)

                assert declaration.variableType == '.b1'

                newDeclaration = PTXDecleration()
                newDeclaration.leadingWhiteSpace = line.leadingWhiteSpace
                newDeclaration.buildString(DeclarationType.Register, None, '.u16', declaration.variableName + '_u16')

                newLoad = PTXFunctionalLine()
                newLoad.leadingWhiteSpace = line.leadingWhiteSpace
                newLoad.buildString('ld.global.u16', (dst + '_u16', ptr))

                newAnd = PTXFunctionalLine()
                newAnd.leadingWhiteSpace = line.leadingWhiteSpace
                newAnd.buildString('and.b16', (dst + '_u16', dst + '_u16', '%const1_u16'))

                newSetp = PTXFunctionalLine()
                newSetp.leadingWhiteSpace = line.leadingWhiteSpace
                newSetp.buildString('setp.eq.u16', (dst, dst + '_u16', '%const1_u16'))


                declaration.buildString(declaration.declarationType, declaration.vector, '.pred', declaration.variableName)
                ptx_shader.lines.remove(line)
                ptx_shader.lines[index:index] = (newDeclaration, newLoad, newAnd, newSetp)
            

def add_consts(ptx_shader):
    const1_u16_Declaration = PTXDecleration()
    const1_u16_Declaration.leadingWhiteSpace = '\t'
    const1_u16_Declaration.buildString(DeclarationType.Register, None, '.u16', '%const1_u16')

    const1_u16_Mov = PTXFunctionalLine()
    const1_u16_Mov.leadingWhiteSpace = '\t'
    const1_u16_Mov.buildString('mov.u16', ('%const1_u16', '1'))

    ptx_shader.addToStart((const1_u16_Declaration, const1_u16_Mov, PTXLine('\n')))



    const0_u32_Declaration = PTXDecleration()
    const0_u32_Declaration.leadingWhiteSpace = '\t'
    const0_u32_Declaration.buildString(DeclarationType.Register, None, '.u32', '%const0_u32')

    const0_u32_Mov = PTXFunctionalLine()
    const0_u32_Mov.leadingWhiteSpace = '\t'
    const0_u32_Mov.buildString('mov.u32', ('%const0_u32', '0'))

    ptx_shader.addToStart((const0_u32_Declaration, const0_u32_Mov, PTXLine('\n')))




def translate_ALU(ptx_shader):
    for index in range(len(ptx_shader.lines)):
        line = ptx_shader.lines[index]
        
        if line.instructionClass != InstructionClass.Functional:
            continue

        if line.functionalType == FunctionalType.fpow:
            dst, src1, src2 = line.args # dst = src1 ^ src2 ?

            declaration, _ = ptx_shader.findDeclaration(dst)
            assert declaration.variableType == '.f32'

            logLine = PTXFunctionalLine()
            logLine.leadingWhiteSpace = line.leadingWhiteSpace
            logLine.buildString('lg2.approx.f32', (dst, src1))

            mulLine = PTXFunctionalLine()
            mulLine.leadingWhiteSpace = line.leadingWhiteSpace
            mulLine.buildString('mul.f32', (dst, dst, src2))

            expLine = PTXFunctionalLine()
            expLine.leadingWhiteSpace = line.leadingWhiteSpace
            expLine.buildString('ex2.approx.f32', (dst, dst))

            ptx_shader.lines.remove(line)
            ptx_shader.lines[index:index] = (logLine, mulLine, expLine)



def main():
    unique_ID = 0
    assert len(sys.argv) == 2
    shaderPath = sys.argv[1]
    shader = PTXShader(shaderPath)
    
    add_consts(shader)

    translate_load_const(shader)
    translate_descriptor_set_instructions(shader)
    translate_deref_instructions(shader)
    translate_trace_ray(shader)
    translate_decl_var(shader)
    translate_load_GL_instructions(shader)
    translate_image_deref_store(shader)
    translate_exit(shader)
    translate_phi(shader)

    translate_vector_operands(shader, unique_ID)
    translate_const_operands(shader)

    translate_ALU(shader)

    translate_f1_to_pred(shader)
    shader.writeToFile(shaderPath)


main()