#include "kernels.cu.h"
#include "../shared_cuda.cu.h"

template<typename MappedFunction>
float multiGpuMapping(
    typename MappedFunction::InputElement* input_array, 
    typename MappedFunction::X x, 
    typename MappedFunction::ReturnElement* output_array, 
    const unsigned long int array_len,
    const bool use_hints=false
) {  
    int origin_device;
    CCC(cudaGetDevice(&origin_device));
    int device_count;
    CCC(cudaGetDeviceCount(&device_count));

    size_t block_count = (array_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t dev_block_count = (block_count + device_count - 1) / device_count;

    unsigned long int per_device = array_len / device_count;
    int remainder = array_len % device_count;
    unsigned long int running_total = 0;
    unsigned long int device_start;
    unsigned long int device_len;

    cudaEvent_t* start_events = 
        (cudaEvent_t*)malloc(device_count * sizeof(cudaEvent_t));
    cudaEvent_t* end_events = 
        (cudaEvent_t*)malloc(device_count * sizeof(cudaEvent_t));

    if (use_hints) {
        for (int device=0; device<device_count; device++) {
            device_start = running_total;
            device_len = (remainder > 0) ? per_device + 1 : per_device;
            remainder -= 1;
            running_total += device_len;

            CCC(cudaMemAdvise(
                input_array+device_start, 
                device_len*sizeof(typename MappedFunction::InputElement), 
                cudaMemAdviseSetPreferredLocation, 
                device
            ));
            CCC(cudaMemAdvise(
                output_array+device_start, 
                device_len*sizeof(typename MappedFunction::ReturnElement), 
                cudaMemAdviseSetPreferredLocation, 
                device
            ));
        }
        running_total = 0;
    }

    setup_events(&start_events, origin_device, device_count);
    setup_events(&end_events, origin_device, device_count);

    for (int device=0; device<device_count; device++) {
        CCC(cudaSetDevice(device));

        device_start = running_total;
        device_len = (remainder > 0) ? per_device + 1 : per_device;
        remainder -= 1;
        running_total += device_len;

        typename MappedFunction::InputElement* sub_input_array = 
            input_array + device_start;
        typename MappedFunction::ReturnElement* sub_output_array = 
            output_array + device_start;

        CCC(cudaEventRecord(start_events[device]));
        mappingKernel<MappedFunction><<<dev_block_count, BLOCK_SIZE>>>(
            sub_input_array, x, sub_output_array, device_len
        );
        CCC(cudaEventRecord(end_events[device]));
    }

    for (int device=0; device<device_count; device++) {
        CCC(cudaSetDevice(device));
        CCC(cudaEventSynchronize(end_events[device]));
    }  

    float runtime_milliseconds = get_mean_runtime(device_count, &start_events, &end_events);

    free(start_events);
    free(end_events);

    CCC(cudaSetDevice(origin_device));

    return runtime_milliseconds * 1e3;
}