/*
 * Time.h
 *
 *  Created on: 27/7/2016
 *      Author: abian
 */

#include <iostream>
#include <chrono>

#ifndef TIME_H_
#define TIME_H_

class Timer {
	typedef std::chrono::high_resolution_clock::time_point time_point;
	typedef std::chrono::duration<double> duration;
protected:
	std::ostream* verbose_out;
	time_point start;

public:
	Timer();
	Timer(std::ostream* verbose_out);
	virtual ~Timer();

	/*!
	 * \brief Getting the current elapsed time value
	 */
	double getElapsedTime();

	/*!
	 * \brief Printing the current elapsed time value
	 */
	void printElapsedTime();
};

#endif /* TIME_H_ */
