from ctypes import *
import win32api
import sys
import os

dll_path = '.\\DMD_dll\\dependency\\ONN_with_DMD_dll.dll'
sys.path.append("C:\\Program Files\\Teledyne\\Spinnaker\\bin64\\vs2015")
img_num = 1

class dmd_funcs:
    def __init__(self):
        self.dmd_dll = windll.LoadLibrary(dll_path)
        self.dmd_handle = c_void_p(None)
        print("Load_DMD_funcs!")
        
    def dmd_init(self):
        self.dmd_dll.sys_init.restype = c_uint64
        self.dmd_handle = self.dmd_dll.sys_init()
        return self
        
    def dmd_deinit(self):
        self.dmd_dll.sys_deinit(c_uint64(self.dmd_handle))
        win32api.FreeLibrary(self.dmd_dll._handle)
    
    def run_dmd(self, input_img, weights, img_num):
        img_ptr = cast(input_img.ctypes.data, POINTER(c_ubyte))
        weight_ptr = cast(weights.ctypes.data, POINTER(c_ubyte))
        ret = self.dmd_dll.run_dmd_sys(c_uint64(self.dmd_handle), img_ptr, weight_ptr, img_num)
        
        return ret
    
    # test funcs
    def test_func(self):
        ret = self.dmd_dll.test_img()
        print('ret = ' + str(ret))
        return ret
    
    def test_img(self, input_img, weights, img_num):
        img_ptr = cast(input_img.ctypes.data, POINTER(c_ubyte))
        weight_ptr = cast(weights.ctypes.data, POINTER(c_ubyte))
        ret = self.dmd_dll.img_buf_test(img_ptr, weight_ptr, img_num)
        return ret
    

# res = dmd_test.test_img()
# dmd_handle = c_void_p(None)

# dmd_test.sys_init.restype = c_uint64
# dmd_handle = dmd_test.sys_init()

# dmd_test.sys_deinit(c_uint64(dmd_handle))


# for i in range(0, img_num) :
#     img_path = './images/Acquisition-C-' + str(i) + '.jpg'
#     img = mpimg.imread(img_path)

# plt.imshow(img)