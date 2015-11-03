#pragma once

#include <memory>
#include <type_traits>

template<class T, class... Ts>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Ts&&... Args) {
    return (std::unique_ptr<T>(new T(std::forward<Ts>(Args)...)));
}

template<class T>
typename std::enable_if<std::is_array<T>::value && std::extent<T>::value == 0, std::unique_ptr<T>>::type
make_unique(size_t size) {
    typedef typename std::remove_extent<T>::type element;
    
    return (std::unique_ptr<T>(new element[size]()));
}

template<class T, class... Ts>
typename std::enable_if<std::extent<T>::value != 0, void>::type
make_unique(Ts&&...) = delete;
