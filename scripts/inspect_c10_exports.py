"""Resolve nearest exported symbol to a locally observed Windows crash RVA."""
from pathlib import Path
import ctypes
import json
import struct

path=Path(__file__).resolve().parents[1]/'.venv/Lib/site-packages/torch/lib/c10.dll'
data=path.read_bytes()
def u16(off):return struct.unpack_from('<H',data,off)[0]
def u32(off):return struct.unpack_from('<I',data,off)[0]
pe=u32(0x3c);sections=u16(pe+6);optional=pe+24;export_rva=u32(optional+112)
section_start=optional+u16(pe+20)
def offset(rva):
    for i in range(sections):
        pos=section_start+i*40;virtual_size=u32(pos+8);virtual_addr=u32(pos+12)
        if virtual_addr<=rva<virtual_addr+max(virtual_size,u32(pos+16)):
            return u32(pos+20)+rva-virtual_addr
    return rva
def string(rva):
    off=offset(rva);return data[off:data.index(b'\x00',off)].decode('ascii')
exports=offset(export_rva);count=u32(exports+24)
funcs=offset(u32(exports+28));names=offset(u32(exports+32));ordinals=offset(u32(exports+36))
entries=[]
undecorate=ctypes.WinDLL('dbghelp').UnDecorateSymbolName
undecorate.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong,ctypes.c_ulong]
for i in range(count):
    name=string(u32(names+4*i));ordinal=u16(ordinals+2*i);rva=u32(funcs+4*ordinal)
    if abs(rva-0x91444)<100000:
        output=ctypes.create_string_buffer(4096);undecorate(name.encode(),output,4096,0)
        entries.append({'rva':hex(rva),'delta_from_fault':0x91444-rva,'symbol':output.value.decode() or name})
entries.sort(key=lambda x:abs(x['delta_from_fault']))
result={'fault_rva':'0x91444','note':'Nearest export is evidence of location only; no PDB/line-level native dump available', 'nearest_exports':entries[:12]}
target=Path(__file__).resolve().parents[1]/'experiments/sequential-ocr-fix/c10-exports.json'
target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
