#pragma once
#pragma once
#include <cstdint>

extern "C" {
void ecg_init();                    // call once after MX_* init
void ecg_on_adc_sample(uint16_t s); // call from HAL_ADC_ConvCpltCallback (1 kHz)
void ecg_poll();                    // call from the main loop
}
