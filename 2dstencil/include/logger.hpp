#pragma once
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdarg>

enum class LogLevel {
	DEBUG = 0,
	INFO = 1,
	WARNING = 2,
	ERROR = 3,
	CRITICAL = 4
};

class Logger {
public:
	static Logger& getInstance() {
		static Logger instance;
		return instance;
	}
	Logger(const Logger&) = delete;
	Logger& operator=(const Logger&) = delete;
	void setLevelEnabled(LogLevel level, bool enabled) {
		levelEnabled[static_cast<int>(level)] = enabled;
	}
	void print(LogLevel level, const std::string& buffer) {
		if (!levelEnabled[static_cast<int>(level)]) {
			return;
		}
		std::string logMessage = "[" + getLevelString(level) + "] " + buffer;
		std::cout << logMessage << std::endl;
	}
	void print(LogLevel level, const char* format, ...) {
		char buffer[1024];
		va_list args;
		va_start(args, format);
		vsnprintf(buffer, sizeof(buffer), format, args);
		va_end(args);
		print(level, std::string(buffer));
	}

private:
	bool levelEnabled[5];
	Logger() {
		levelEnabled[static_cast<int>(LogLevel::DEBUG)] = true;
		levelEnabled[static_cast<int>(LogLevel::INFO)] = true;
		levelEnabled[static_cast<int>(LogLevel::WARNING)] = true;
		levelEnabled[static_cast<int>(LogLevel::ERROR)] = true;
		levelEnabled[static_cast<int>(LogLevel::CRITICAL)] = true;
	}
	std::string getLevelString(LogLevel level) {
		switch(level) {
			case LogLevel::DEBUG:    return "DEBUG";
			case LogLevel::INFO:     return "INFO";
			case LogLevel::WARNING:  return "WARNING";
			case LogLevel::ERROR:    return "ERROR";
			case LogLevel::CRITICAL: return "CRITICAL";
			default: return "UNKNOWN";
		}
	}
};

#define LOG_DEBUG(format, ...)    Logger::getInstance().print(LogLevel::DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...)     Logger::getInstance().print(LogLevel::INFO, format, ##__VA_ARGS__)
#define LOG_WARNING(format, ...)  Logger::getInstance().print(LogLevel::WARNING, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...)    Logger::getInstance().print(LogLevel::ERROR, format, ##__VA_ARGS__)
#define LOG_CRITICAL(format, ...) Logger::getInstance().print(LogLevel::CRITICAL, format, ##__VA_ARGS__)
