#pragma once

#include <iostream>
#include <sstream>
#include <type_traits>

class Logger {
public:
	template <typename Arg>
	void log(Arg&& a) {
		data << a << "\n";
	}

	template <typename Arg, typename... Args>
	void log(Arg&& a, Args&&... args) {
		data << a << ";";
		log(std::forward<Args>(args)...);
	}

	void writeTo(std::ostream& str) {
		str << data.str();
	}

private:
	std::ostringstream data;
};

