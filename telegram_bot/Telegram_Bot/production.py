from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ConversationHandler,
    CallbackContext, filters, CallbackQueryHandler
)
from datetime import datetime, timezone
from enum import Enum, auto
from typing import List, Optional
from mongodb import productions_collection, users_collection, operations_collection


class Genre(Enum):
    ENTERTAINMENT = auto()
    EDUCATIONAL = auto()
    GAME = auto()
    PRODUCTIVITY = auto()
    SOCIAL_MEDIA = auto()
    UTILITY = auto()


class CoType(Enum):
    CPC = auto()
    CPA = auto()
    CPT = auto()
    Guaranteed = auto()


class Channel(Enum):
    TENCENT = auto()
    XIAOHONGSHU = auto()


class State(Enum):
    ACTIVATED = auto()
    UNACTIVATED = auto()


class Production:
    def __init__(
            self,
            name: str,
            groupID: int,
            url: str,
            co_type: CoType,
            channel: Channel,
            genre: Genre,
            CPA_price: float = 0.00,
            CPC_price: float = 0.00,
            CPT_price: float = 0.00,
            quota: float = 0.00,  # 余额
            start_date: Optional[datetime] = None,
            principal: Optional[List[str]] = None,
            group_principals: Optional[List[str]] = None,
            backstage_link: Optional[str] = None,
            backstage_account: Optional[str] = None,
            backstage_password: Optional[str] = None,
            remark: Optional[str] = None,
            business_acc: Optional[str] = None,
            last_update_date: Optional[datetime] = None,
            state: State = State.ACTIVATED
    ):
        self.name = name
        self.groupID = groupID
        self.url = url
        self.co_type = co_type
        self.channel = channel
        self.genres = genre
        self.CPA_price = CPA_price
        self.CPC_price = CPC_price
        self.CPT_price = CPT_price
        self.start_date = start_date if start_date is not None else datetime.today()
        self.principal = principal if principal is not None else []
        self.group_principals = group_principals if group_principals is not None else []
        self.backstage_link = backstage_link
        self.backstage_account = backstage_account
        self.backstage_password = backstage_password
        self.remark = remark
        self.business_acc = business_acc
        self.quota = quota
        self.last_update_date = last_update_date if last_update_date is not None else datetime.now(timezone.utc)
        self.state = state


(
    PRODUCTION_NAME,
    PRODUCTION_URL,
    PRODUCTION_COTYPE,
    PRODUCTION_CHANNEL,
    PRODUCTION_GENRE,
    PRODUCTION_CPA_PRICE,
    PRODUCTION_CPC_PRICE,
    PRODUCTION_CPT_PRICE,
    PRODUCTION_QUOTA,
    PRODUCTION_PRINCIPAL,
    PRODUCTION_GROUP_MEMBERS,
    PRODUCTION_START_DATE,
    PRODUCTION_BACKSTAGE_LINK,
    PRODUCTION_BACKSTAGE_ACCOUNT,
    PRODUCTION_BACKSTAGE_PASSWORD,
    PRODUCTION_REMARK,
    PRODUCTION_BUSINESS_ACCOUNT,
    SAVE_PRODUCTION,
    SELECT_PRODUCTION,
    SELECT_OPERATION,
    ENTER_AMOUNT
) = range(21)


async def log_operation(user_id, group_id, username, operation_type, details):
    operation_data = {
        'timestamp': datetime.now(timezone.utc),
        'group_id': group_id,
        'user_id': user_id,
        'username': username,
        'operation_type': operation_type,
        'details': details
    }
    await operations_collection.insert_one(operation_data)


############################################################################################创建产品
async def start_create_production(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    user_role = await fetch_user_role(user_id)

    print(user_role)

    if user_role == 'staff':
        await update.message.reply_text("请输入产品名称:")
        return PRODUCTION_NAME
    else:
        await update.message.reply_text("您不是员工，无权创建。")
        return ConversationHandler.END


async def input_production_name(update: Update, context: CallbackContext) -> int:
    context.user_data['name'] = update.message.text
    group_id = update.message.chat_id
    print(group_id)
    context.user_data['groupID'] = group_id
    await update.message.reply_text("请输入产品URL:")
    return PRODUCTION_URL


async def input_production_url(update: Update, context: CallbackContext) -> int:
    context.user_data['url'] = update.message.text
    await update.message.reply_text("请输入产品合作方式(CPC, CPA, CPT, Guaranteed):")
    return PRODUCTION_COTYPE


async def input_production_coType(update: Update, context: CallbackContext) -> int:
    text = update.message.text.upper()
    try:
        selected_co_type = CoType[text]
        context.user_data['co_type'] = selected_co_type
        await update.message.reply_text("请输入产品渠道 (例如：TENCENT, XIAOHONGSHU):")
        # test
        return PRODUCTION_CHANNEL

    except KeyError:
        await update.message.reply_text(
            "输入无效。请选择下列其中之一: " + ", ".join([e.name for e in CoType]),
            reply_markup=ReplyKeyboardMarkup(
                [[e.name for e in CoType]], one_time_keyboard=True
            )
        )
        return PRODUCTION_COTYPE


async def input_production_channel(update: Update, context: CallbackContext) -> int:
    context.user_data['channel'] = update.message.text
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(genre.name, callback_data=genre.name) for genre in Genre]

    ])
    ##
    try:
        for genre in Genre:
            print(genre)
    except:
        pass
    ##
    await update.message.reply_text("请选择产品的类型:", reply_markup=keyboard)
    return PRODUCTION_GENRE


async def genre_button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    genre_data = query.data

    context.user_data['genre'] = genre_data
    print(genre_data)

    await query.edit_message_text(text="请输入产品CPA单价:")

    return PRODUCTION_CPA_PRICE


async def input_production_cpa_price(update: Update, context: CallbackContext) -> int:
    cpa_price = update.message.text
    context.user_data['cpa_price'] = float(cpa_price)
    await update.message.reply_text("请输入产品CPC单价:")
    return PRODUCTION_CPC_PRICE


async def input_production_cpc_price(update: Update, context: CallbackContext) -> int:
    cpc_price = update.message.text
    context.user_data['cpc_price'] = float(cpc_price)
    await update.message.reply_text("请输入产品CPT单价:")
    return PRODUCTION_CPT_PRICE


async def input_production_cpt_price(update: Update, context: CallbackContext) -> int:
    cpt_price = update.message.text
    context.user_data['cpt_price'] = float(cpt_price)
    await update.message.reply_text("请输入余额:")

    return PRODUCTION_QUOTA


async def input_production_quota(update: Update, context: CallbackContext) -> int:
    quota = update.message.text
    context.user_data['quota'] = float(quota)
    print(f"Quota captured: {quota}")  # Debug print

    try:
        staff_members = await fetch_staff_members(users_collection)
        print("Staff Members:", staff_members)

        if not staff_members:
            await update.message.reply_text("目前没有任何员工。")
            return ConversationHandler.END

        # 创建负责人选择按钮
        keyboard = [[InlineKeyboardButton(member.get('username', 'No Username') or 'No Username',
                                          callback_data=str(member['_id']))]
                    for member in staff_members]

        # 添加完成选择的按钮
        keyboard.append([InlineKeyboardButton('完成选择', callback_data='confirm_selection')])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text('请选择产品的负责人:', reply_markup=reply_markup)

        context.user_data['staff_members'] = staff_members
        context.user_data['selected_principals'] = []

    except Exception as e:
        await update.message.reply_text(f"查询负责人时出错: {e}")
        return ConversationHandler.END

    return PRODUCTION_PRINCIPAL


async def principal_button_callback(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    data = query.data

    # 用户选择或取消选择一个员工负责人
    selected_principals = context.user_data.get('selected_principals', [])
    if data in selected_principals:
        selected_principals.remove(data)
    else:
        selected_principals.append(data)

    context.user_data['selected_principals'] = selected_principals
    selected_names = [member['username'] or 'No Username' for member in context.user_data['staff_members'] if
                      str(member['_id']) in selected_principals]

    # 如果用户点击完成选择
    if data == 'confirm_selection':
        if not selected_principals:
            await query.edit_message_text(text='您还没有选择负责人。')
            return PRODUCTION_PRINCIPAL

        await query.edit_message_text(text=f'您选择了负责人：{", ".join(selected_names)}。')

        # 将选择的负责人用户名存储到context
        context.user_data['principal_usernames'] = selected_names

        # 提示用户输入产品负责人的姓名
        await query.message.reply_text('请输入产品负责人的姓名（多个姓名用逗号分隔）:')
        return PRODUCTION_GROUP_MEMBERS

    # 重新创建按钮列表，保留当前选择状态
    keyboard = [
        [InlineKeyboardButton(member.get('username', 'No Username') or 'No Username', callback_data=str(member['_id']))]
        for member in context.user_data['staff_members']]
    keyboard.append([InlineKeyboardButton('完成选择', callback_data='confirm_selection')])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(text=f"当前选择的负责人有：{', '.join(selected_names)}。点击 '完成选择' 继续。",
                                  reply_markup=reply_markup)

    return PRODUCTION_PRINCIPAL


async def group_members_input_callback(update: Update, context: CallbackContext) -> int:
    context.user_data['group_principals'] = update.message.text
    # 显示选择的产品负责人并提示输入产品开始日期
    await update.message.reply_text(f"您选择了产品负责人：{context.user_data['group_principals']}。请输入产品开始的日期 (格式：YYYY-MM-DD):")
    return PRODUCTION_START_DATE



async def input_production_start_date(update: Update, context: CallbackContext) -> int:
    text = update.message.text
    try:
        # 转换为date格式
        start_date = datetime.strptime(text, '%Y-%m-%d')
        context.user_data['start_date'] = start_date
        await update.message.reply_text(
            f"开始日期已设置为: {start_date.strftime('%Y-%m-%d')}"
        )

        await update.message.reply_text("请输入后台链接:")
        return PRODUCTION_BACKSTAGE_LINK
    except ValueError:
        # 如果转换失败，通知用户并重新请求输入
        await update.message.reply_text(
            "日期格式不正确，请使用 YYYY-MM-DD 格式重新输入日期。"
        )
        return PRODUCTION_START_DATE


async def input_production_backstage_link(update: Update, context: CallbackContext) -> int:
    context.user_data['backstage_link'] = update.message.text
    await update.message.reply_text("请输入后台账号:")
    return PRODUCTION_BACKSTAGE_ACCOUNT


async def input_production_backstage_account(update: Update, context: CallbackContext) -> int:
    context.user_data['backstage_account'] = update.message.text
    await update.message.reply_text("请输入后台密码:")
    return PRODUCTION_BACKSTAGE_PASSWORD


async def input_production_backstage_password(update: Update, context: CallbackContext) -> int:
    context.user_data['backstage_password'] = update.message.text
    await update.message.reply_text("请输入备注信息:")
    return PRODUCTION_REMARK


async def input_production_remark(update: Update, context: CallbackContext) -> int:
    remark = update.message.text
    context.user_data['remark'] = remark
    await update.message.reply_text("请输入打款和收款的商业账户或钱包地址:")
    return PRODUCTION_BUSINESS_ACCOUNT


async def input_production_business_account(update: Update, context: CallbackContext) -> int:
    business_acc = update.message.text
    context.user_data['business_acc'] = business_acc
    await update.message.reply_text(f"您输入的商业账户是：{business_acc}。请确认无误后任意输入进行保存。")
    return SAVE_PRODUCTION


##保存##
async def save_production(update: Update, context: CallbackContext) -> int:
    production_data = {
        'name': context.user_data['name'],
        'groupID': context.user_data['groupID'],
        'url': context.user_data['url'],
        'co_type': context.user_data['co_type'].value,
        'channel': context.user_data['channel'],
        'genre': context.user_data['genre'],
        'CPA_price': context.user_data['cpa_price'],
        'CPC_price': context.user_data['cpc_price'],
        'CPT_price': context.user_data['cpt_price'],
        'quota': context.user_data['quota'],
        'principal': context.user_data['principal_usernames'],
        'group_principals': context.user_data['group_principals'],
        'start_date': context.user_data['start_date'],
        'remark': context.user_data['remark'],
        'business_acc': context.user_data['business_acc'],
        'state': State,
    }
    production = Production(**production_data)
    # 保存到MongoDB
    result = await productions_collection.insert_one(production.__dict__)  # 是否异步？
    chat_id = update.effective_chat.id if update.effective_chat else update.callback_query.message.chat_id

    user = update.effective_user
    username = user.username if user.username else 'No Username'
    await log_operation(user.id, chat_id, username, 'Create Production', {'production_name': production_data['name']})

    await context.bot.send_message(chat_id=chat_id, text=f"产品已创建，ID: {result.inserted_id}.")
    return ConversationHandler.END


async def cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("创建过程已取消。", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


################################################################################################
#########查询Production#######

# async def list_production(update: Update, context: CallbackContext) -> int:


#################数据库接口####################  PRODUCTION  #######################


# 根据groupid找回production
async def find_production_by_groupid(id):
    find_production = productions_collection.find({'groupID': id})
    production = await find_production.to_list(length=100)
    return production


async def list_productions_by_groupid(update: Update, context: CallbackContext) -> None:
    group_id = update.effective_chat.id
    productions = await find_production_by_groupid(group_id)
    if not productions:
        await update.message.reply_text('没有找到任何产品信息。')
        return

    message_text = '产品列表:\n'
    for index, prod in enumerate(productions, start=1):
        message_text += (
            f"{index}. 名称: {prod['name']}\n"
            f"   URL: {prod['url']}\n"
            f"   合作方式: {CoType(prod['co_type']).name if isinstance(prod['co_type'], int) else prod['co_type']}\n"
            f"   渠道: {prod['channel']}\n"
            f"   类型: {prod['genres']}\n"
            f"   CPA价格: {prod['CPA_price']}\n"
            f"   CPC价格: {prod['CPC_price']}\n"
            f"   CPT价格: {prod['CPT_price']}\n"
            f"   余额: {prod['quota']}\n"
            f"   负责人: {', '.join(prod['principal'])}\n"
            f"   产品负责人: {prod['group_principals']}\n"
            f"   备注: {prod.get('remark', '无')}\n"

            "------\n"
        )

    messages = [message_text[i:i + 4096] for i in range(0, len(message_text), 4096)]
    for message in messages:
        await update.message.reply_text(message)


#################数据库接口####################  USER  #######################
async def fetch_user_role(user_id):
    user_document = await users_collection.find_one({'user_id': user_id})
    if user_document:
        return user_document.get('role')
    else:
        return None


# 找回员工
async def fetch_staff_members(users_collection):
    cursor = users_collection.find({'role': 'staff'})
    staff_members = await cursor.to_list(length=100)
    return staff_members


SETTLE_PRODUCTION, SETTLE_QUANTITY = range(2)


async def start_settle_production(update: Update, context: CallbackContext) -> int:
    group_id = update.effective_chat.id
    productions = await find_production_by_groupid(group_id)
    if not productions:
        await update.message.reply_text('没有找到任何产品信息。')
        return ConversationHandler.END

    message_text = '产品列表:\n'
    for index, prod in enumerate(productions, start=1):
        message_text += (
            f"{index}. 名称: {prod['name']}\n"
            f"   余额: {prod['quota']}\n"
            "------\n"
        )

    messages = [message_text[i:i + 4096] for i in range(0, len(message_text), 4096)]
    for message in messages:
        await update.message.reply_text(message)

    await update.message.reply_text("请输入要结算的产品名称:")
    return SETTLE_PRODUCTION


async def get_production_id(update: Update, context: CallbackContext) -> int:
    context.user_data['production_name'] = update.message.text
    await update.message.reply_text("请输入要结算的数量:")
    return SETTLE_QUANTITY


async def settle_production(update: Update, context: CallbackContext) -> int:
    production_name = context.user_data['production_name']
    quantity = float(update.message.text)

    production = await productions_collection.find_one({'name': production_name})
    if not production:
        await update.message.reply_text("找不到该产品。")
        return ConversationHandler.END

    # 根据产品的合作方式选择相应的价格
    co_type = production['co_type']
    if isinstance(co_type, int):  # 如果是整数，转换为 CoType 枚举
        co_type = CoType(co_type).name
    elif isinstance(co_type, str):
        co_type = co_type.upper()
    else:
        await update.message.reply_text("无法识别的合作方式。")
        return ConversationHandler.END

    if co_type == 'CPC':
        price = production['CPC_price']
    elif co_type == 'CPA':
        price = production['CPA_price']
    elif co_type == 'CPT':
        price = production['CPT_price']
    else:
        await update.message.reply_text("无法识别的合作方式。")
        return ConversationHandler.END

    total_cost = quantity * price

    if production['quota'] < total_cost:
        await update.message.reply_text("余额不足，需要充值。")
        return ConversationHandler.END

    new_quota = production['quota'] - total_cost
    await productions_collection.update_one(
        {'name': production_name},
        {'$set': {'quota': new_quota}}
    )

    await update.message.reply_text(f"结算成功，新的余额为：{new_quota}")
    return ConversationHandler.END


# 操作产品余额的相关函数

async def start_balance_operation(update: Update, context: CallbackContext) -> int:
    group_id = update.effective_chat.id
    productions = await find_production_by_groupid(group_id)
    if not productions:
        await update.message.reply_text('没有找到任何产品信息。')
        return ConversationHandler.END

    message_text = '产品列表:\n'
    for index, prod in enumerate(productions, start=1):
        message_text += (
            f"{index}. 名称: {prod['name']}\n"
            f"   余额: {prod['quota']}\n"
            "------\n"
        )

    messages = [message_text[i:i + 4096] for i in range(0, len(message_text), 4096)]
    for message in messages:
        await update.message.reply_text(message)

    await update.message.reply_text("请输入要操作余额的产品名称:")
    return SELECT_PRODUCTION


async def select_operation(update: Update, context: CallbackContext) -> int:
    context.user_data['production_name'] = update.message.text
    keyboard = [
        [InlineKeyboardButton("充值", callback_data='deposit')],
        [InlineKeyboardButton("提现", callback_data='withdraw')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("请选择要进行的操作:", reply_markup=reply_markup)
    return SELECT_OPERATION


async def handle_operation_choice(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()
    operation = query.data
    context.user_data['operation'] = operation

    if operation == 'deposit':
        await query.edit_message_text(text="请输入充值金额:")
    elif operation == 'withdraw':
        await query.edit_message_text(text="请输入提现金额:")

    return ENTER_AMOUNT


async def enter_amount(update: Update, context: CallbackContext) -> int:
    amount = float(update.message.text)
    operation = context.user_data['operation']
    production_name = context.user_data['production_name']

    production = await productions_collection.find_one({'name': production_name})
    if not production:
        await update.message.reply_text("找不到该产品。")
        return ConversationHandler.END

    if operation == 'deposit':
        new_quota = production['quota'] + amount
    elif operation == 'withdraw':
        if production['quota'] < amount:
            await update.message.reply_text("余额不足，无法提现。")
            return ConversationHandler.END
        new_quota = production['quota'] - amount

    await productions_collection.update_one(
        {'name': production_name},
        {'$set': {'quota': new_quota}}
    )

    user = update.effective_user
    username = user.username if user.username else 'No Username'
    details = {'production_name': production_name, 'amount': amount}
    await log_operation(user.id, update.effective_chat.id, username, operation.capitalize(), details)

    await update.message.reply_text(f"操作成功，新的余额为：{new_quota}")
    return ConversationHandler.END


async def view_logs(update: Update, context: CallbackContext) -> None:
    logs = operations_collection.find().sort('timestamp', -1).limit(10)  # 获取最近10条日志
    logs_list = await logs.to_list(length=10)

    if not logs_list:
        await update.message.reply_text("没有找到操作日志。")
        return

    message_text = "操作日志:\n"
    for log in logs_list:
        message_text += (
            f"时间: {log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"群组ID: {log['group_id']}\n"
            f"员工: {log['username']}\n"
            f"操作: {log['operation_type']}\n"
            f"详情: {log['details']}\n"
            "------\n"
        )

    await update.message.reply_text(message_text)


balance_conv_handler = ConversationHandler(
    entry_points=[CommandHandler('balance', start_balance_operation)],
    states={
        SELECT_PRODUCTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_operation)],
        SELECT_OPERATION: [CallbackQueryHandler(handle_operation_choice)],
        ENTER_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, enter_amount)],
    },
    fallbacks=[CommandHandler('cancel', cancel)],
)

settle_conv_handler = ConversationHandler(
    entry_points=[CommandHandler('settle', start_settle_production)],
    states={
        SETTLE_PRODUCTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_production_id)],
        SETTLE_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, settle_production)],
    },
    fallbacks=[CommandHandler('cancel', cancel)],
)

list_productions_handler = CommandHandler('list', list_productions_by_groupid)

productions_conv_handler = ConversationHandler(
    entry_points=[CommandHandler('create', start_create_production)],
    states={
        PRODUCTION_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_name)],
        PRODUCTION_URL: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_url)],
        PRODUCTION_COTYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_coType)],
        PRODUCTION_CHANNEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_channel)],
        PRODUCTION_GENRE: [
            CallbackQueryHandler(genre_button_callback, pattern='|'.join([genre.name for genre in Genre]))],
        PRODUCTION_CPA_PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_cpa_price)],
        PRODUCTION_CPC_PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_cpc_price)],
        PRODUCTION_CPT_PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_cpt_price)],
        PRODUCTION_QUOTA: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_quota)],
        PRODUCTION_PRINCIPAL: [CallbackQueryHandler(principal_button_callback)],
        PRODUCTION_GROUP_MEMBERS: [MessageHandler(filters.TEXT & ~filters.COMMAND, group_members_input_callback)],
        PRODUCTION_START_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_start_date)],
        PRODUCTION_BACKSTAGE_LINK: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_backstage_link)],
        PRODUCTION_BACKSTAGE_ACCOUNT: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_backstage_account)],
        PRODUCTION_BACKSTAGE_PASSWORD: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_backstage_password)],
        PRODUCTION_REMARK: [MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_remark)],
        PRODUCTION_BUSINESS_ACCOUNT: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, input_production_business_account)],
        SAVE_PRODUCTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_production)],
    },
    fallbacks=[CommandHandler('cancel', cancel)],
)

view_logs_handler = CommandHandler('view_logs', view_logs)
