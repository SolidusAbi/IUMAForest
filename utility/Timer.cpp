/*
 * Time.cpp
 *
 *  Created on: 27/7/2016
 *      Author: Abian Hernandez Guedes
 */

#include <iomanip>

#include "Timer.h"

Timer::Timer() {
	this->verbose_out = &std::cout;
	this->start = std::chrono::high_resolution_clock::now();
}

Timer::Timer(std::ostream* verbose_out){
	this->verbose_out = verbose_out;
	this->start = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {}

double Timer::getElapsedTime(){
	time_point end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::duration<double>>(end - this->start).count();
}

void Timer::printElapsedTime(){
	*verbose_out << "Elapsed time is " << std::fixed << this->getElapsedTime() << " seconds." << std::endl;
}
