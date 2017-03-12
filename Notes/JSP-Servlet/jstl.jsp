JSTL,标准标签类库集合，主要有5个类库，本次笔记记录Core类库和功能（函数）类库


1.Core类库
声明使用Core类库
<%@ taglib uri="http://java.sum.com/jsp/jstl.core" prefix="c" %>

1.1 通用动作指令
<c:out value="value" default="defaultValue" />

<c:set var="varName" value="value" [scope="{page|request|session|application}"] />
<c:set target="target" property="propertyName" value="value" />

<c:remove var="varName" [scope="{page|request|session|application}"] />

1.2 条件式动作指令

1.2.1 if标签
<c:if test="testCondition(一般是EL)">
	主体内容
</c:if >

1.2.2 choose、when、otherwise标签
<c:choose>
	<c:when test="${param.status==null}">
		You are a full number
	</c:when>
	<c:when test="${param.status=='student'}">
		You are a student
	</c:when>
	<c:otherwise>
		Please register
		</c:otherwise>
		</c:choose>

1.2.3 iterator动作指令

forEach标签：

迭代数组、列表
<c:forEach items="${requestScope.books}" var="book">
	${book.name}<br/>
	${book.price}
	</c:forEach>

迭代map
<c:forEach items="${requestScope.capitals}" var="mapItem">
	${mapItem.key} : ${mapItem.value}<br/>
</c:forEach>

forTokens标签
<c:forTokens items="stringOfTokens" delims="delimiters" [var="varName"] [varStatus="statusName"]>
	主体内容
</c:forTokens>

2.功能（函数）类库

声明使用函数类库
<%@ taglib uri="http://java.sum.com/jsp/jstl/functions" prefix="fn" %>

调用函数格式（要用EL调用）
${fn:functionName}

常用函数：

boolean contains(string, substring)
例子：
<c:set var="myString" value="Hello World" />
${fn:contains(myString,"Hello")}
结果为true

boolean containsIgnoreCase(string, substring)

boolean endWith(string,string)

boolean startWith(string, string)

void escapeXml(String)

int indexOf(string, substring) 找不到时返回-1

String join(array, seperator)

int length(string)

String replace(string, beforeString, afterString)

String[] split(string, seperator)

String substring(string, beginIndex, endIndex)

String substringAfter(string, substring) 返回子字符串后的字符串部分

String substringBefore(string, substring) 返回子字符串前的字符串部分

String toLowerCase(string)

String toUpperCase(string)

String trim(string)




