/**
 *  @file error.h
 *  @brief Device Error Management definition
 *  @author Rodolfo Lima
 *  @date February, 2011
 */

#ifndef ERROR_H
#define ERROR_H

//== INCLUDES =================================================================

#include <sstream>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

//== IMPLEMENTATION ===========================================================

/**
 *  @ingroup utils
 *  @brief Check error in device
 *
 *  This function checks if there is a device error.
 *
 *  @param[in] msg Message to appear in case of a device error
 */
inline void cuda_error( const std::string &msg ) {
    cudaError_t err = cudaGetLastError();
    if( err != cudaSuccess ) {
        if( msg.empty() )
            throw std::runtime_error(cudaGetErrorString(err));
        else {
            std::stringstream ss;
            ss << msg << ": " << cudaGetErrorString(err);
            throw std::runtime_error(ss.str().c_str());
        }
    }
}

//=============================================================================
#endif // ERROR_H
//=============================================================================
//vi: ai sw=4 ts=4
